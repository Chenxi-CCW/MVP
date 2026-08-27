from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp.model import MVPPruner, extract_model_config
from mvp.data import MVPCollator, MVPOracleDataset, load_samples_json, normalize_samples, stratified_split_samples
from mvp.question_embedder import LlavaQuestionEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MVPPruner with BCE and coarse/fine base-rank loss.")
    parser.add_argument("--samples-json", type=str, required=True)
    parser.add_argument("--val-samples-json", type=str, default=None)
    parser.add_argument("--llava-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--output-dir", type=str, default="outputs/mvp_pruner")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embedder-dtype", type=str, default="float16")
    parser.add_argument("--amp-dtype", type=str, choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=str, default="none")
    parser.add_argument("--loss-bce-weight", type=float, default=1.0)
    parser.add_argument("--loss-rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-delta", type=float, default=1e-6)
    parser.add_argument("--coarse-bin-size", type=int, default=32)
    parser.add_argument("--coarse-num-bins", type=int, default=6)
    parser.add_argument("--coarse-rank-weight", type=float, default=1.0)
    parser.add_argument("--fine-rank-weight", type=float, default=1.0)
    parser.add_argument("--model-config", type=json.loads, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: str, amp_dtype: str):
    enabled = device != "cpu" and amp_dtype != "none"
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def estimate_soft_pos_weight(samples: list[dict[str, Any]]) -> float:
    pos = 0.0
    neg = 0.0
    for sample in samples:
        mask = np.load(sample["oracle_mask_path"], mmap_mode="r")
        pos += float(mask.sum())
        neg += float(mask.size - mask.sum())
    if pos <= 0.0:
        raise ValueError("No positive label mass found.")
    return neg / pos


class CoarseFineBaseRankLoss(nn.Module):
    def __init__(
        self,
        *,
        delta: float = 1e-6,
        coarse_bin_size: int = 32,
        coarse_num_bins: int = 6,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.delta = float(delta)
        self.coarse_bin_size = int(coarse_bin_size)
        self.coarse_num_bins = int(coarse_num_bins)
        self.coarse_weight = float(coarse_weight)
        self.fine_weight = float(fine_weight)
        self.eps = float(eps)

    def _base_rank(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_gap = targets[:, None] - targets[None, :]
        valid = target_gap > self.delta
        if not torch.any(valid):
            return logits.new_zeros(())
        logit_gap = logits[:, None] - logits[None, :]
        weights = target_gap[valid]
        loss = F.softplus(-logit_gap[valid])
        return (weights * loss).sum() / weights.sum().clamp_min(self.eps)

    def _coarse_rank(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        max_rank = min(self.coarse_bin_size * self.coarse_num_bins, int(targets.numel()))
        if max_rank < self.coarse_bin_size * 2:
            return logits.new_zeros(())
        sorted_idx = torch.argsort(targets, descending=True)[:max_rank]
        bins = sorted_idx.reshape(-1, self.coarse_bin_size)
        bin_logits = logits[bins].mean(dim=1)
        bin_targets = targets[bins].mean(dim=1)
        return self._base_rank(bin_logits, bin_targets)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.to(torch.float32)
        targets = targets.to(torch.float32)
        losses = []
        for sample_logits, sample_targets in zip(logits, targets):
            fine = self._base_rank(sample_logits, sample_targets)
            coarse = self._coarse_rank(sample_logits, sample_targets)
            denom = max(self.fine_weight + self.coarse_weight, self.eps)
            losses.append((self.fine_weight * fine + self.coarse_weight * coarse) / denom)
        if not losses:
            return logits.new_zeros(())
        return torch.stack(losses).mean()


class MVPTrainCriterion(nn.Module):
    def __init__(
        self,
        *,
        bce_weight: float,
        rank_weight: float,
        rank_loss: nn.Module,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.rank_weight = float(rank_weight)
        self.rank_loss = rank_loss
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        targets = targets.to(torch.float32)
        bce = F.binary_cross_entropy_with_logits(logits.to(torch.float32), targets, pos_weight=self.pos_weight)
        rank = self.rank_loss(logits, targets)
        total = self.bce_weight * bce + self.rank_weight * rank
        return {"total": total, "bce": bce, "rank": rank}


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        **batch,
        "visual_tokens": batch["visual_tokens"].to(device, non_blocking=True),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "oracle_mask": batch["oracle_mask"].to(device, non_blocking=True),
    }


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, loss_dict: dict[str, torch.Tensor], threshold: float) -> torch.Tensor:
    probs = torch.sigmoid(logits.detach())
    preds = (probs >= threshold).to(torch.float32)
    hard_targets = (targets.detach() >= threshold).to(torch.float32)
    tp = ((preds == 1) & (hard_targets == 1)).sum()
    fp = ((preds == 1) & (hard_targets == 0)).sum()
    fn = ((preds == 0) & (hard_targets == 1)).sum()
    sample_count = torch.tensor(float(targets.shape[0]), device=targets.device, dtype=torch.float64)
    return torch.stack(
        [
            loss_dict["total"].detach().to(torch.float64) * sample_count,
            loss_dict["bce"].detach().to(torch.float64) * sample_count,
            loss_dict["rank"].detach().to(torch.float64) * sample_count,
            tp.to(torch.float64),
            fp.to(torch.float64),
            fn.to(torch.float64),
            probs.sum().to(torch.float64),
            targets.detach().sum().to(torch.float64),
            torch.tensor(float(targets.numel()), device=targets.device, dtype=torch.float64),
            sample_count,
        ]
    )


def summarize_metrics(stats: torch.Tensor) -> dict[str, float]:
    sample_count = max(float(stats[9].item()), 1.0)
    tp, fp, fn = float(stats[3].item()), float(stats[4].item()), float(stats[5].item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    numel = max(float(stats[8].item()), 1.0)
    return {
        "loss": float(stats[0].item()) / sample_count,
        "bce": float(stats[1].item()) / sample_count,
        "rank": float(stats[2].item()) / sample_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_keep_ratio": float(stats[6].item()) / numel,
        "soft_keep_ratio": float(stats[7].item()) / numel,
    }


def run_epoch(
    *,
    model: MVPPruner,
    question_embedder: LlavaQuestionEmbedder,
    loader: DataLoader,
    criterion: MVPTrainCriterion,
    optimizer: AdamW | None,
    scheduler,
    device: str,
    amp_dtype: str,
    threshold: float,
    max_grad_norm: float,
    log_every: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stats = torch.zeros(10, device=device, dtype=torch.float64)
    progress = tqdm(loader, leave=False, dynamic_ncols=True)

    for step, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        with torch.no_grad():
            question_embeds = question_embedder(batch["input_ids"])

        with torch.set_grad_enabled(is_train), autocast_context(device, amp_dtype):
            logits = model(
                visual_tokens=batch["visual_tokens"],
                question_tokens=question_embeds,
                question_attention_mask=batch["attention_mask"],
            )
            loss_dict = criterion(logits, batch["oracle_mask"])

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        stats += batch_metrics(logits, batch["oracle_mask"], loss_dict, threshold)
        if step % max(log_every, 1) == 0 or step == len(loader):
            partial = summarize_metrics(stats)
            progress.set_description(("train" if is_train else "val") + f" loss={partial['loss']:.4f} f1={partial['f1']:.4f}")

    return summarize_metrics(stats)


def save_checkpoint(path: Path, *, model: MVPPruner, optimizer: AdamW, scheduler, epoch: int, best_metric: float, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
            "metrics": metrics,
            "model_config": model.model_config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, meta, train_root = load_samples_json(args.samples_json)
    samples = normalize_samples(samples, train_root=train_root)
    if args.val_samples_json:
        train_samples = samples
        val_samples, _, val_root = load_samples_json(args.val_samples_json)
        val_samples = normalize_samples(val_samples, train_root=val_root)
    else:
        train_samples, val_samples = stratified_split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)

    question_embedder = LlavaQuestionEmbedder(args.llava_path, device=device, dtype=args.embedder_dtype)
    collator = MVPCollator(pad_token_id=question_embedder.pad_token_id)
    train_loader = DataLoader(
        MVPOracleDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device != "cpu",
        collate_fn=collator,
    )
    val_loader = DataLoader(
        MVPOracleDataset(val_samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device != "cpu",
        collate_fn=collator,
    )

    model = MVPPruner.from_config(extract_model_config(args.model_config)).to(device)
    pos_weight_value = estimate_soft_pos_weight(train_samples) if args.pos_weight == "auto" else None
    if args.pos_weight not in {"auto", "none"}:
        pos_weight_value = float(args.pos_weight)
    pos_weight = None if pos_weight_value is None else torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    criterion = MVPTrainCriterion(
        bce_weight=args.loss_bce_weight,
        rank_weight=args.loss_rank_weight,
        rank_loss=CoarseFineBaseRankLoss(
            delta=args.rank_delta,
            coarse_bin_size=args.coarse_bin_size,
            coarse_num_bins=args.coarse_num_bins,
            coarse_weight=args.coarse_rank_weight,
            fine_weight=args.fine_rank_weight,
        ),
        pos_weight=pos_weight,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = max(len(train_loader) * args.epochs, 1)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates)

    config = {
        **vars(args),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "samples_json_meta": meta,
        "pos_weight": pos_weight_value,
        "model_config": model.model_config,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    history: list[dict[str, Any]] = []
    best_val_f1 = -math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            question_embedder=question_embedder,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp_dtype=args.amp_dtype,
            threshold=args.threshold,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                question_embedder=question_embedder,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                device=device,
                amp_dtype=args.amp_dtype,
                threshold=args.threshold,
                max_grad_norm=args.max_grad_norm,
                log_every=args.log_every,
            )

        row = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        print(json.dumps(row, ensure_ascii=False))
        history.append(row)
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_val_f1, args=args, metrics=row)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_val_f1, args=args, metrics=row)
        if args.save_every_epoch:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_val_f1, args=args, metrics=row)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
