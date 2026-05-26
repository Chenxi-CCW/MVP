from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def load_samples_json(samples_json: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    path = Path(samples_json).resolve()
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        samples = list(obj.get("samples", []))
        meta = {k: v for k, v in obj.items() if k != "samples"}
        train_root = Path(obj.get("train_dir", path.parent.parent)).resolve()
    elif isinstance(obj, list):
        samples = obj
        meta = {}
        train_root = path.parent.resolve()
    else:
        raise TypeError(f"Unsupported JSON root type: {type(obj).__name__}")

    return samples, meta, train_root


def _resolve_path(sample: dict[str, Any], field: str, train_root: Path, fallback: str | None = None) -> str:
    if field in sample and sample[field]:
        candidate = Path(sample[field])
        return str(candidate.resolve() if candidate.is_absolute() else (train_root / candidate).resolve())
    if fallback is not None:
        return str((train_root / fallback.format(sample_id=sample["sample_id"])).resolve())
    raise KeyError(f"Missing path field: {field}")


def normalize_samples(samples: list[dict[str, Any]], train_root: Path) -> list[dict[str, Any]]:
    normalized = []
    for sample in samples:
        item = dict(sample)
        item["hidden_path"] = _resolve_path(
            sample=item,
            field="hidden_path",
            train_root=train_root,
            fallback=item.get("hidden_rel", "hidden/{sample_id}.npy"),
        )
        item["question_tokens_path"] = _resolve_path(
            sample=item,
            field="question_tokens_path",
            train_root=train_root,
            fallback=item.get("question_tokens_rel", "question_tokens/{sample_id}.npz"),
        )
        item["oracle_mask_path"] = _resolve_path(
            sample=item,
            field="oracle_mask_path",
            train_root=train_root,
            fallback=(
                item.get("soft_mask_rel")
                or item.get("final_mask_rel")
                or "masks/oracle_min/{sample_id}.npy"
            ),
        )
        normalized.append(item)
    return normalized


def stratified_split_samples(
    samples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(str(sample.get("subset", "default")), []).append(sample)

    rng = random.Random(seed)
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []

    for group_samples in groups.values():
        group_samples = list(group_samples)
        rng.shuffle(group_samples)
        n_val = max(1, int(math.ceil(len(group_samples) * val_ratio)))
        if n_val >= len(group_samples):
            n_val = max(1, len(group_samples) - 1)
        val_samples.extend(group_samples[:n_val])
        train_samples.extend(group_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


class QVTSOracleDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], sample_weights: list[float] | None = None):
        self.samples = samples
        if sample_weights is not None and len(sample_weights) != len(samples):
            raise ValueError("sample_weights length must match samples length.")
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        visual_tokens = np.load(sample["hidden_path"]).astype(np.float32, copy=False)
        if visual_tokens.shape == (577, 1024):
            visual_tokens = visual_tokens[1:]
        if visual_tokens.shape != (576, 1024):
            raise ValueError(
                f"Unexpected visual token shape {visual_tokens.shape} for {sample['sample_id']}"
            )

        question_tokens = np.load(sample["question_tokens_path"])
        input_ids = question_tokens["input_ids"].astype(np.int64, copy=False)
        attention_mask = question_tokens["attention_mask"].astype(np.int64, copy=False)
        if input_ids.ndim != 1 or attention_mask.ndim != 1:
            raise ValueError(f"Question tokens must be 1D for {sample['sample_id']}")
        if len(input_ids) != len(attention_mask):
            raise ValueError(f"Question token length mismatch for {sample['sample_id']}")

        oracle_mask = np.load(sample["oracle_mask_path"]).astype(np.float32, copy=False)
        if oracle_mask.shape != (576,):
            raise ValueError(
                f"Unexpected oracle mask shape {oracle_mask.shape} for {sample['sample_id']}"
            )

        return {
            "sample_id": sample["sample_id"],
            "subset": str(sample.get("subset", "unknown")),
            "question": str(sample.get("question", "")),
            "sample_weight": torch.tensor(
                1.0 if self.sample_weights is None else float(self.sample_weights[index]),
                dtype=torch.float32,
            ),
            "visual_tokens": torch.from_numpy(visual_tokens),
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "oracle_mask": torch.from_numpy(oracle_mask),
        }


class QVTSCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "subsets": [item["subset"] for item in batch],
            "questions": [item["question"] for item in batch],
            "sample_weight": torch.stack([item["sample_weight"] for item in batch], dim=0),
            "visual_tokens": torch.stack([item["visual_tokens"] for item in batch], dim=0),
            "input_ids": pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "oracle_mask": torch.stack([item["oracle_mask"] for item in batch], dim=0),
        }


def estimate_pos_weight(samples: list[dict[str, Any]]) -> float:
    pos = 0.0
    neg = 0.0
    for sample in samples:
        mask = np.load(sample["oracle_mask_path"], mmap_mode="r")
        pos += float(mask.sum())
        neg += float(mask.size - mask.sum())
    if pos <= 0:
        raise ValueError("No positive labels found in oracle masks.")
    return neg / pos
