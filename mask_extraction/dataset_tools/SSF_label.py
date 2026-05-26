from __future__ import annotations

"""
SSF_label.py

训练集 Success-Failure Family-Beta Soft Label 构建：
  - 读取候选方法：
      grasp
      vispruner_64
      vispruner_128
      scope_64
      scope_128
  - 先基于 train_inference/*.jsonl 统计每个 mask key 的：
      - success rate
      - average keep ratio
      - evidence efficiency = acc / (avg_keep_ratio + eps)
  - 在 family 内将 evidence efficiency 归一化为 alpha_m
  - 再计算 family 级可靠性 beta_f：
      - 使用 family max efficiency
  - 对每个样本：
      - 若所有候选方法都答错，则丢弃并写入 hard_failure_pool
      - 在每个 family 内构造：
          p_f = weighted-average(correct masks by alpha)
          q_f = weighted-average(wrong masks by alpha)
      - 跨 family 构造：
          P = weighted-average(p_f by beta) over active positive families
          Q = weighted-average(q_f by beta) over active failure families
      - 最终 soft label：
          y = P * (1 - Q * (1 - P))
      - Acc/R/E/alpha/beta 按 subset 分开统计，并在样本上优先使用对应 subset 的统计量
      - target_keep_ratio 继续沿用 oracle-min correct mask

输出：
  soft_masks/{fusion_key}/{sample_id}.npy
  records/{fusion_key}.jsonl
  train_supervision/{fusion_key}_samples.json
  train_supervision/{fusion_key}_decisions.jsonl
  train_supervision/{fusion_key}_summary.json
  train_supervision/{fusion_key}_hard_failure_pool.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve()
_DATASET_TOOLS = _HERE.parent
_PROJECT_ROOT = _HERE.parents[2]
for _p in [_DATASET_TOOLS, _PROJECT_ROOT]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mask_extraction.dataset_tools.train_data_side_utils import load_final_samples_json


DEFAULT_MASK_KEYS = [
    "grasp",
    "scope_64",
    "scope_128",
    "vispruner_64",
    "vispruner_128",
]

METHOD_PRIORITY = {
    "grasp": 0,
    "sam": 0,
    "vispruner": 1,
    "scope": 2,
}

FAMILY_TO_MASK_KEYS = {
    "grasp": ["grasp", "sam"],
    "vispruner": ["vispruner_64", "vispruner_128"],
    "scope": ["scope_64", "scope_128"],
}

MASK_SIZE = 576


def _infer_method(mask_key: str) -> str:
    if mask_key in {"grasp", "sam"} or mask_key.startswith(("grasp_", "sam_")):
        return "grasp"
    if mask_key.startswith("scope"):
        return "scope"
    if mask_key.startswith("vispruner"):
        return "vispruner"
    return mask_key.split("_", 1)[0]


def _build_selected_family_to_mask_keys(mask_keys: list[str]) -> dict[str, list[str]]:
    selected = set(mask_keys)
    return {
        family_name: [mask_key for mask_key in family_mask_keys if mask_key in selected]
        for family_name, family_mask_keys in FAMILY_TO_MASK_KEYS.items()
    }


def _build_inference_keys(mask_keys: list[str], run_prefix: str, inference_keys: list[str] | None) -> list[str]:
    if inference_keys is not None:
        if len(inference_keys) != len(mask_keys):
            raise ValueError("inference_keys 与 mask_keys 长度必须一致。")
        return inference_keys
    return [f"{run_prefix}{k}" for k in mask_keys]


def _load_candidates(
    train_dir: Path,
    sample_ids: set[str],
    inference_keys: list[str],
    mask_keys: list[str],
) -> dict[str, list[dict[str, Any]]]:
    candidates_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for order_idx, (inf_key, mask_key) in enumerate(zip(inference_keys, mask_keys)):
        path = train_dir / "train_inference" / f"{inf_key}.jsonl"
        if not path.exists():
            print(f"  [WARN] inference file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                sample_id = str(rec.get("sample_id", ""))
                if not sample_id or sample_id not in sample_ids:
                    continue

                candidates_by_sample[sample_id].append(
                    {
                        "sample_id": sample_id,
                        "subset": str(rec.get("subset", "")),
                        "mask_key": mask_key,
                        "source_run": inf_key,
                        "method": _infer_method(mask_key),
                        "order_idx": order_idx,
                        "correct": rec.get("correct") is True,
                        "n_kept": int(rec.get("n_kept", 0)),
                        "keep_ratio": float(rec.get("keep_ratio", 0.0)),
                        "output": str(rec.get("output", "")),
                    }
                )

    return candidates_by_sample


def _sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda x: int(x["order_idx"]))


def _candidate_rank(cand: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(cand["n_kept"]),
        METHOD_PRIORITY.get(str(cand["method"]), 999),
        int(cand["order_idx"]),
    )


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = np.load(mask_path).astype(bool, copy=False)
    if mask.shape != (MASK_SIZE,):
        raise ValueError(f"unexpected mask shape {mask.shape}")
    return mask


def _format_float_for_name(value: float) -> str:
    return format(float(value), "g")


def _resolve_fusion_key(
    fusion_key: str | None,
) -> str:
    if fusion_key is not None and str(fusion_key).strip():
        return str(fusion_key)
    return "ssf_family_beta_soft_label"


def _init_mask_stat_buckets(mask_keys: list[str]) -> dict[str, dict[str, float | int | str]]:
    return {
        key: {
            "mask_key": key,
            "family": _infer_method(key),
            "num_present": 0,
            "num_correct": 0,
            "sum_keep_ratio": 0.0,
        }
        for key in mask_keys
    }


def _finalize_mask_statistics(
    stats: dict[str, dict[str, float | int | str]],
    family_to_mask_keys: dict[str, list[str]],
    epsilon: float,
) -> dict[str, dict[str, float | int | str]]:
    family_peak_eff: dict[str, float] = {}
    for family_name, family_mask_keys in family_to_mask_keys.items():
        max_eff = 0.0
        for mask_key in family_mask_keys:
            item = stats[mask_key]
            num_present = int(item["num_present"])
            num_correct = int(item["num_correct"])
            avg_keep_ratio = (
                float(item["sum_keep_ratio"]) / num_present
                if num_present > 0
                else 0.0
            )
            acc = num_correct / num_present if num_present > 0 else 0.0
            efficiency = acc / (avg_keep_ratio + epsilon) if num_present > 0 else 0.0
            item["acc"] = float(acc)
            item["avg_keep_ratio"] = float(avg_keep_ratio)
            item["evidence_efficiency"] = float(efficiency)
            max_eff = max(max_eff, float(efficiency))
        family_peak_eff[family_name] = max_eff

    for family_name, family_mask_keys in family_to_mask_keys.items():
        family_max_eff = family_peak_eff[family_name]
        denom = family_max_eff + epsilon
        for mask_key in family_mask_keys:
            efficiency = float(stats[mask_key]["evidence_efficiency"])
            alpha = efficiency / denom if family_max_eff > 0.0 else 0.0
            stats[mask_key]["alpha"] = float(alpha)
            stats[mask_key]["family_max_evidence_efficiency"] = float(family_max_eff)

    return stats


def _compute_family_statistics_from_mask_stats(
    mask_stats: dict[str, dict[str, float | int | str]],
    family_to_mask_keys: dict[str, list[str]],
    epsilon: float,
) -> dict[str, dict[str, float | int | str]]:
    family_stats: dict[str, dict[str, float | int | str]] = {}
    global_family_score = 0.0

    for family_name, family_mask_keys in family_to_mask_keys.items():
        eff_values = [float(mask_stats[mask_key]["evidence_efficiency"]) for mask_key in family_mask_keys]
        family_eff = float(max(eff_values)) if eff_values else 0.0
        family_stats[family_name] = {
            "family": family_name,
            "mask_keys": list(family_mask_keys),
            "num_mask_keys": len(family_mask_keys),
            "beta_reduce": "max",
            "family_evidence_efficiency": float(family_eff),
        }
        global_family_score = max(global_family_score, family_eff)

    for family_name in family_stats:
        family_eff = float(family_stats[family_name]["family_evidence_efficiency"])
        beta = family_eff / (global_family_score + epsilon) if global_family_score > 0.0 else 0.0
        family_stats[family_name]["beta"] = float(beta)
        family_stats[family_name]["global_max_family_evidence_efficiency"] = float(global_family_score)

    return family_stats


def _compute_mask_and_family_statistics(
    candidates_by_sample: dict[str, list[dict[str, Any]]],
    mask_keys: list[str],
    family_to_mask_keys: dict[str, list[str]],
    epsilon: float,
) -> tuple[
    dict[str, dict[str, float | int | str]],
    dict[str, dict[str, float | int | str]],
    dict[str, dict[str, dict[str, float | int | str]]],
    dict[str, dict[str, dict[str, float | int | str]]],
]:
    global_stats = _init_mask_stat_buckets(mask_keys)
    subset_stats_raw: dict[str, dict[str, dict[str, float | int | str]]] = {}

    for sample_candidates in candidates_by_sample.values():
        for cand in sample_candidates:
            mask_key = str(cand["mask_key"])
            subset = str(cand.get("subset", ""))
            if mask_key not in global_stats:
                continue
            global_stats[mask_key]["num_present"] += 1
            global_stats[mask_key]["num_correct"] += int(bool(cand["correct"]))
            global_stats[mask_key]["sum_keep_ratio"] += float(cand["keep_ratio"])

            if subset not in subset_stats_raw:
                subset_stats_raw[subset] = _init_mask_stat_buckets(mask_keys)
            subset_stats_raw[subset][mask_key]["num_present"] += 1
            subset_stats_raw[subset][mask_key]["num_correct"] += int(bool(cand["correct"]))
            subset_stats_raw[subset][mask_key]["sum_keep_ratio"] += float(cand["keep_ratio"])

    global_stats = _finalize_mask_statistics(global_stats, family_to_mask_keys, epsilon)
    global_family_stats = _compute_family_statistics_from_mask_stats(global_stats, family_to_mask_keys, epsilon)

    subset_mask_stats = {
        subset: _finalize_mask_statistics(stats, family_to_mask_keys, epsilon)
        for subset, stats in subset_stats_raw.items()
    }
    subset_family_stats = {
        subset: _compute_family_statistics_from_mask_stats(stats, family_to_mask_keys, epsilon)
        for subset, stats in subset_mask_stats.items()
    }

    return global_stats, global_family_stats, subset_mask_stats, subset_family_stats


def _weighted_binary_mask_average(
    mask_items: list[tuple[float, np.ndarray]],
    *,
    epsilon: float,
) -> np.ndarray:
    if not mask_items:
        return np.zeros((MASK_SIZE,), dtype=np.float32)

    weighted_sum = np.zeros((MASK_SIZE,), dtype=np.float32)
    total_weight = 0.0
    for weight, mask in mask_items:
        weight = float(weight)
        if weight <= 0.0:
            continue
        weighted_sum += weight * mask.astype(np.float32, copy=False)
        total_weight += weight
    if total_weight <= 0.0:
        return np.zeros((MASK_SIZE,), dtype=np.float32)
    return (weighted_sum / (total_weight + epsilon)).astype(np.float32, copy=False)


def _weighted_dense_support_average(
    items: list[tuple[float, np.ndarray]],
    *,
    epsilon: float,
) -> np.ndarray:
    if not items:
        return np.zeros((MASK_SIZE,), dtype=np.float32)

    weighted_sum = np.zeros((MASK_SIZE,), dtype=np.float32)
    total_weight = 0.0
    for weight, support in items:
        weight = float(weight)
        if weight <= 0.0:
            continue
        weighted_sum += weight * np.clip(support.astype(np.float32, copy=False), 0.0, 1.0)
        total_weight += weight
    if total_weight <= 0.0:
        return np.zeros((MASK_SIZE,), dtype=np.float32)
    return (weighted_sum / (total_weight + epsilon)).astype(np.float32, copy=False)


def _build_family_evidence(
    *,
    family_name: str,
    family_loaded_keys: list[str],
    loaded_masks_by_key: dict[str, np.ndarray],
    loaded_correct_by_key: dict[str, bool],
    mask_stats: dict[str, dict[str, float | int | str]],
    family_stats: dict[str, dict[str, float | int | str]],
    epsilon: float,
) -> dict[str, Any]:
    correct_keys = [key for key in family_loaded_keys if loaded_correct_by_key[key]]
    wrong_keys = [key for key in family_loaded_keys if not loaded_correct_by_key[key]]

    positive_items = [
        (float(mask_stats[key]["alpha"]), loaded_masks_by_key[key])
        for key in correct_keys
    ]
    failure_items = [
        (float(mask_stats[key]["alpha"]), loaded_masks_by_key[key])
        for key in wrong_keys
    ]

    p = _weighted_binary_mask_average(positive_items, epsilon=epsilon)
    q = _weighted_binary_mask_average(failure_items, epsilon=epsilon)
    beta = float(family_stats[family_name]["beta"])

    return {
        "family_name": family_name,
        "loaded_keys": family_loaded_keys,
        "correct_keys": correct_keys,
        "wrong_keys": wrong_keys,
        "active_positive": bool(correct_keys),
        "beta": beta,
        "positive_support": p.astype(np.float32, copy=False),
        "failure_support": q.astype(np.float32, copy=False),
        "positive_support_keep_ratio": float(p.mean()),
        "failure_support_keep_ratio": float(q.mean()),
        "alpha_by_key": {key: round(float(mask_stats[key]["alpha"]), 6) for key in family_loaded_keys},
        "efficiency_by_key": {
            key: round(float(mask_stats[key]["evidence_efficiency"]), 6)
            for key in family_loaded_keys
        },
    }


def run_build_train_soft_labels_success_failure_family_beta(
    train_dir: str,
    samples_json: str,
    mask_keys: list[str],
    run_prefix: str,
    inference_keys: list[str] | None,
    fusion_key: str | None,
    output_root_name: str,
    epsilon: float,
) -> None:
    root = Path(train_dir)
    if not root.exists():
        raise FileNotFoundError(f"train_dir not found: {root}")

    samples, input_summary = load_final_samples_json(samples_json)
    sample_ids = {str(s["sample_id"]) for s in samples}

    family_to_mask_keys = _build_selected_family_to_mask_keys(mask_keys)
    inference_keys = _build_inference_keys(mask_keys, run_prefix, inference_keys)
    candidates_by_sample = _load_candidates(root, sample_ids, inference_keys, mask_keys)
    global_mask_stats, global_family_stats, subset_mask_stats, subset_family_stats = _compute_mask_and_family_statistics(
        candidates_by_sample,
        mask_keys,
        family_to_mask_keys=family_to_mask_keys,
        epsilon=epsilon,
    )
    fusion_key = _resolve_fusion_key(fusion_key)

    soft_mask_out_dir = root / output_root_name / fusion_key
    records_path = root / "records" / f"{fusion_key}.jsonl"
    supervision_dir = root / "train_supervision"
    samples_path = supervision_dir / f"{fusion_key}_samples.json"
    decisions_path = supervision_dir / f"{fusion_key}_decisions.jsonl"
    summary_path = supervision_dir / f"{fusion_key}_summary.json"
    hard_failure_pool_path = supervision_dir / f"{fusion_key}_hard_failure_pool.jsonl"

    soft_mask_out_dir.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    supervision_dir.mkdir(parents=True, exist_ok=True)

    output_samples: list[dict[str, Any]] = []
    summary_by_subset: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "retained": 0, "dropped": 0, "failed": 0}
    )
    n_correct_dist = Counter()
    n_wrong_dist = Counter()
    n_loaded_family_dist = Counter()
    n_active_family_dist = Counter()
    correct_method_combo_dist = Counter()
    wrong_method_combo_dist = Counter()
    loaded_family_combo_dist = Counter()
    active_family_combo_dist = Counter()
    winning_method_dist = Counter()
    winning_method_family_dist = Counter()
    dropped_no_correct_count = 0
    missing_inference_count = 0
    failure_reasons = Counter()
    soft_keep_ratios: list[float] = []
    target_keep_ratios: list[float] = []
    global_positive_keep_ratios: list[float] = []
    global_failure_keep_ratios: list[float] = []
    written_sample_ids: set[str] = set()

    with open(records_path, "w", encoding="utf-8") as f_records, open(
        decisions_path, "w", encoding="utf-8"
    ) as f_decisions, open(hard_failure_pool_path, "w", encoding="utf-8") as f_hard_failure:
        for sample in samples:
            sample_id = str(sample["sample_id"])
            subset = str(sample["subset"])
            summary_by_subset[subset]["total"] += 1

            all_cands = _sort_candidates(list(candidates_by_sample.get(sample_id, [])))
            correct_cands = [c for c in all_cands if c["correct"]]
            wrong_cands = [c for c in all_cands if not c["correct"]]

            correct_methods = [str(c["mask_key"]) for c in correct_cands]
            wrong_methods = [str(c["mask_key"]) for c in wrong_cands]
            correct_families = [str(c["method"]) for c in correct_cands]
            wrong_families = [str(c["method"]) for c in wrong_cands]

            n_correct = len(correct_cands)
            n_wrong = len(wrong_cands)
            n_correct_dist[n_correct] += 1
            n_wrong_dist[n_wrong] += 1

            decision: dict[str, Any] = {
                "uid": str(sample["uid"]),
                "orig_id": str(sample["orig_id"]),
                "sample_id": sample_id,
                "subset": subset,
                "retained": False,
                "n_candidates": len(all_cands),
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "correct_methods": correct_methods,
                "wrong_methods": wrong_methods,
                "correct_families": correct_families,
                "wrong_families": wrong_families,
                "candidate_scores": [
                    {
                        "mask_key": c["mask_key"],
                        "family": c["method"],
                        "correct": bool(c["correct"]),
                        "n_kept": int(c["n_kept"]),
                        "keep_ratio": float(c["keep_ratio"]),
                    }
                    for c in all_cands
                ],
            }

            if not all_cands:
                missing_inference_count += 1
                failure_reasons["missing_inference"] += 1
                summary_by_subset[subset]["failed"] += 1
                decision["failure_reason"] = "missing_inference"
                f_decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                continue

            if n_correct == 0:
                dropped_no_correct_count += 1
                summary_by_subset[subset]["dropped"] += 1
                decision["drop_reason"] = "no_correct_method"
                decision["failure_reason"] = None
                hard_failure_entry = {
                    **decision,
                    "hard_failure_reason": "all_candidate_masks_failed",
                }
                f_hard_failure.write(json.dumps(hard_failure_entry, ensure_ascii=False) + "\n")
                f_decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                continue

            loaded_masks_by_key: dict[str, np.ndarray] = {}
            loaded_correct_by_key: dict[str, bool] = {}
            valid_correct_cands: list[dict[str, Any]] = []
            valid_wrong_cands: list[dict[str, Any]] = []
            missing_mask_keys: list[str] = []

            try:
                for cand in all_cands:
                    mask_path = root / "masks" / cand["mask_key"] / f"{sample_id}.npy"
                    if not mask_path.exists():
                        missing_mask_keys.append(str(cand["mask_key"]))
                        continue

                    mask = _load_mask(mask_path)
                    loaded_masks_by_key[str(cand["mask_key"])] = mask
                    loaded_correct_by_key[str(cand["mask_key"])] = bool(cand["correct"])

                    cand = dict(cand)
                    cand["mask_path"] = mask_path
                    if cand["correct"]:
                        valid_correct_cands.append(cand)
                    else:
                        valid_wrong_cands.append(cand)
            except Exception as exc:
                failure_reasons["mask_load_failed"] += 1
                summary_by_subset[subset]["failed"] += 1
                decision["missing_mask_keys"] = missing_mask_keys
                decision["failure_reason"] = f"mask_load_failed:{exc}"
                f_decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                continue

            decision["missing_mask_keys"] = missing_mask_keys

            if not valid_correct_cands:
                failure_reasons["correct_mask_missing"] += 1
                summary_by_subset[subset]["failed"] += 1
                decision["failure_reason"] = "correct_mask_missing"
                f_decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                continue

            mask_stats = subset_mask_stats.get(subset, global_mask_stats)
            family_stats = subset_family_stats.get(subset, global_family_stats)

            family_info_by_name: dict[str, dict[str, Any]] = {}
            family_source_keys: dict[str, list[str]] = {}
            family_source_runs: dict[str, list[str]] = {}
            family_correct_counts: dict[str, int] = {}
            family_wrong_counts: dict[str, int] = {}
            family_positive_keep_ratios: dict[str, float] = {}
            family_failure_keep_ratios: dict[str, float] = {}
            family_alpha_by_key: dict[str, dict[str, float]] = {}
            family_efficiency_by_key: dict[str, dict[str, float]] = {}
            family_beta_by_name: dict[str, float] = {}

            for family_name, family_mask_keys in family_to_mask_keys.items():
                family_loaded_keys = [key for key in family_mask_keys if key in loaded_masks_by_key]
                if not family_loaded_keys:
                    continue

                family_info = _build_family_evidence(
                    family_name=family_name,
                    family_loaded_keys=family_loaded_keys,
                    loaded_masks_by_key=loaded_masks_by_key,
                    loaded_correct_by_key=loaded_correct_by_key,
                    mask_stats=mask_stats,
                    family_stats=family_stats,
                    epsilon=epsilon,
                )
                family_info_by_name[family_name] = family_info
                family_source_keys[family_name] = family_loaded_keys
                family_source_runs[family_name] = [
                    str(c["source_run"])
                    for c in all_cands
                    if c["mask_key"] in family_loaded_keys
                ]
                family_correct_counts[family_name] = len(family_info["correct_keys"])
                family_wrong_counts[family_name] = len(family_info["wrong_keys"])
                family_positive_keep_ratios[family_name] = family_info["positive_support_keep_ratio"]
                family_failure_keep_ratios[family_name] = family_info["failure_support_keep_ratio"]
                family_alpha_by_key[family_name] = family_info["alpha_by_key"]
                family_efficiency_by_key[family_name] = family_info["efficiency_by_key"]
                family_beta_by_name[family_name] = float(family_info["beta"])

            if not family_info_by_name:
                failure_reasons["loaded_family_empty"] += 1
                summary_by_subset[subset]["failed"] += 1
                decision["failure_reason"] = "loaded_family_empty"
                f_decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                continue

            loaded_families = [family for family in family_to_mask_keys if family in family_info_by_name]
            active_families = [
                family
                for family in loaded_families
                if bool(family_info_by_name[family]["active_positive"])
            ]
            failure_active_families = [
                family
                for family in loaded_families
                if bool(family_info_by_name[family]["wrong_keys"])
            ]

            n_loaded_family_dist[len(loaded_families)] += 1
            n_active_family_dist[len(active_families)] += 1

            if not active_families:
                failure_reasons["active_positive_family_empty"] += 1
                summary_by_subset[subset]["failed"] += 1
                decision["failure_reason"] = "active_positive_family_empty"
                f_decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                continue

            global_positive_support = _weighted_dense_support_average(
                [
                    (
                        family_info_by_name[family]["beta"],
                        family_info_by_name[family]["positive_support"],
                    )
                    for family in active_families
                ],
                epsilon=epsilon,
            )
            global_failure_support = _weighted_dense_support_average(
                [
                    (
                        family_info_by_name[family]["beta"],
                        family_info_by_name[family]["failure_support"],
                    )
                    for family in loaded_families
                    if family in failure_active_families
                ],
                epsilon=epsilon,
            )
            soft_mask = global_positive_support * (
                1.0 - global_failure_support * (1.0 - global_positive_support)
            )
            soft_mask = np.clip(soft_mask, 0.0, 1.0).astype(np.float32, copy=False)

            evidence = np.logical_or(global_positive_support > 0.0, global_failure_support > 0.0)
            positive_evidence = global_positive_support > 0.0
            fusion_mode = "failure_aware_family_beta_weighted_average_soft_evidence_distillation"

            winning = min(correct_cands, key=_candidate_rank)
            winning_method = str(winning["mask_key"])
            winning_method_family = str(winning["method"])
            winning_run = str(winning["source_run"])
            winning_output = str(winning["output"])

            winning_loaded_mask = loaded_masks_by_key.get(winning_method)
            if winning_loaded_mask is not None:
                target_n_kept = int(winning_loaded_mask.sum())
                target_keep_ratio = float(winning_loaded_mask.mean())
            else:
                target_n_kept = int(winning["n_kept"])
                target_keep_ratio = float(winning["keep_ratio"])

            soft_keep_ratio = float(soft_mask.mean())
            global_positive_keep_ratio = float(global_positive_support.mean())
            global_failure_keep_ratio = float(global_failure_support.mean())
            out_soft_mask_path = soft_mask_out_dir / f"{sample_id}.npy"
            np.save(out_soft_mask_path, soft_mask)
            written_sample_ids.add(sample_id)

            if correct_methods:
                correct_method_combo_dist["&".join(correct_methods)] += 1
            if wrong_methods:
                wrong_method_combo_dist["&".join(wrong_methods)] += 1
            if loaded_families:
                loaded_family_combo_dist["&".join(loaded_families)] += 1
            if active_families:
                active_family_combo_dist["&".join(active_families)] += 1
            winning_method_dist[winning_method] += 1
            winning_method_family_dist[winning_method_family] += 1

            record = {
                "uid": str(sample["uid"]),
                "orig_id": str(sample["orig_id"]),
                "sample_id": sample_id,
                "task_name": "train",
                "doc_id": sample_id,
                "subset": subset,
                "question": str(sample["question"]),
                "ground_truth": str(sample["answer"]),
                "method": "ssf_family_beta_soft_label",
                "fusion_mode": fusion_mode,
                "epsilon": epsilon,
                "family_beta_reduce": "max",
                "reliability_subset": subset if subset in subset_mask_stats else "__global__",
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "correct_methods": correct_methods,
                "wrong_methods": wrong_methods,
                "correct_families": correct_families,
                "wrong_families": wrong_families,
                "loaded_families": loaded_families,
                "active_families": active_families,
                "family_source_keys": family_source_keys,
                "family_source_runs": family_source_runs,
                "family_correct_counts": family_correct_counts,
                "family_wrong_counts": family_wrong_counts,
                "family_beta_by_name": {k: round(v, 6) for k, v in family_beta_by_name.items()},
                "family_positive_keep_ratios": {k: round(v, 4) for k, v in family_positive_keep_ratios.items()},
                "family_failure_keep_ratios": {k: round(v, 4) for k, v in family_failure_keep_ratios.items()},
                "family_alpha_by_key": family_alpha_by_key,
                "family_efficiency_by_key": family_efficiency_by_key,
                "global_positive_keep_ratio": round(global_positive_keep_ratio, 4),
                "global_failure_keep_ratio": round(global_failure_keep_ratio, 4),
                "winning_method": winning_method,
                "winning_method_family": winning_method_family,
                "winning_run": winning_run,
                "winning_n_kept": target_n_kept,
                "winning_keep_ratio": round(float(target_keep_ratio), 4),
                "winning_output": winning_output,
                "winning_mask_exists": winning_loaded_mask is not None,
                "soft_keep_ratio": round(soft_keep_ratio, 4),
                "n_evidence_tokens": int(evidence.sum()),
                "n_positive_evidence_tokens": int(positive_evidence.sum()),
                "target_n_kept": target_n_kept,
                "target_keep_ratio": round(float(target_keep_ratio), 4),
                "target_ratio_source_method": winning_method,
                "target_ratio_source_family": winning_method_family,
                "used_wrong_mask_count": len(valid_wrong_cands),
                "used_correct_mask_count": len(valid_correct_cands),
                "has_valid_wrong": bool(valid_wrong_cands),
            }
            f_records.write(json.dumps(record, ensure_ascii=False) + "\n")

            sample_entry = dict(sample)
            sample_entry.update(
                {
                    "soft_mask_key": fusion_key,
                    "soft_mask_rel": f"{output_root_name}/{fusion_key}/{sample_id}.npy",
                    "hidden_rel": f"hidden/{sample_id}.npy",
                    "question_tokens_rel": f"question_tokens/{sample_id}.npz",
                    "text_tokens_rel": f"question_tokens/{sample_id}.npz",
                    "n_correct": n_correct,
                    "n_wrong": n_wrong,
                    "correct_methods": correct_methods,
                    "wrong_methods": wrong_methods,
                    "correct_families": correct_families,
                    "wrong_families": wrong_families,
                    "loaded_families": loaded_families,
                    "active_families": active_families,
                    "family_source_keys": family_source_keys,
                    "family_source_runs": family_source_runs,
                    "family_correct_counts": family_correct_counts,
                    "family_wrong_counts": family_wrong_counts,
                    "family_beta_by_name": {k: round(v, 6) for k, v in family_beta_by_name.items()},
                    "family_positive_keep_ratios": {k: round(v, 4) for k, v in family_positive_keep_ratios.items()},
                    "family_failure_keep_ratios": {k: round(v, 4) for k, v in family_failure_keep_ratios.items()},
                    "family_alpha_by_key": family_alpha_by_key,
                    "family_efficiency_by_key": family_efficiency_by_key,
                    "global_positive_keep_ratio": round(global_positive_keep_ratio, 4),
                    "global_failure_keep_ratio": round(global_failure_keep_ratio, 4),
                    "winning_method": winning_method,
                    "winning_method_family": winning_method_family,
                    "winning_run": winning_run,
                    "winning_n_kept": target_n_kept,
                    "winning_keep_ratio": round(float(target_keep_ratio), 4),
                    "winning_output": winning_output,
                    "winning_mask_exists": winning_loaded_mask is not None,
                    "soft_keep_ratio": round(soft_keep_ratio, 4),
                    "target_n_kept": target_n_kept,
                    "target_keep_ratio": round(float(target_keep_ratio), 4),
                    "target_ratio_source_method": winning_method,
                    "target_ratio_source_family": winning_method_family,
                    "target_ratio_source_mask_rel": f"masks/{winning_method}/{sample_id}.npy",
                    "fusion_mode": fusion_mode,
                    "epsilon": epsilon,
                    "family_beta_reduce": "max",
                    "reliability_subset": subset if subset in subset_mask_stats else "__global__",
                    "used_wrong_mask_count": len(valid_wrong_cands),
                    "used_correct_mask_count": len(valid_correct_cands),
                    "has_valid_wrong": bool(valid_wrong_cands),
                    "n_evidence_tokens": int(evidence.sum()),
                    "n_positive_evidence_tokens": int(positive_evidence.sum()),
                }
            )
            output_samples.append(sample_entry)

            decision.update(
                {
                    "retained": True,
                    "failure_reason": None,
                    "drop_reason": None,
                    "fusion_mode": fusion_mode,
                    "epsilon": epsilon,
                    "family_beta_reduce": "max",
                    "reliability_subset": subset if subset in subset_mask_stats else "__global__",
                    "loaded_families": loaded_families,
                    "active_families": active_families,
                    "family_source_keys": family_source_keys,
                    "family_source_runs": family_source_runs,
                    "family_correct_counts": family_correct_counts,
                    "family_wrong_counts": family_wrong_counts,
                    "family_beta_by_name": {k: round(v, 6) for k, v in family_beta_by_name.items()},
                    "family_positive_keep_ratios": {k: round(v, 4) for k, v in family_positive_keep_ratios.items()},
                    "family_failure_keep_ratios": {k: round(v, 4) for k, v in family_failure_keep_ratios.items()},
                    "family_alpha_by_key": family_alpha_by_key,
                    "family_efficiency_by_key": family_efficiency_by_key,
                    "global_positive_keep_ratio": round(global_positive_keep_ratio, 4),
                    "global_failure_keep_ratio": round(global_failure_keep_ratio, 4),
                    "winning_method": winning_method,
                    "winning_method_family": winning_method_family,
                    "winning_run": winning_run,
                    "winning_n_kept": target_n_kept,
                    "winning_keep_ratio": round(float(target_keep_ratio), 4),
                    "winning_mask_exists": winning_loaded_mask is not None,
                    "soft_keep_ratio": round(soft_keep_ratio, 4),
                    "target_n_kept": target_n_kept,
                    "target_keep_ratio": round(float(target_keep_ratio), 4),
                    "target_ratio_source_method": winning_method,
                    "target_ratio_source_family": winning_method_family,
                    "used_wrong_mask_count": len(valid_wrong_cands),
                    "used_correct_mask_count": len(valid_correct_cands),
                    "has_valid_wrong": bool(valid_wrong_cands),
                    "n_evidence_tokens": int(evidence.sum()),
                    "n_positive_evidence_tokens": int(positive_evidence.sum()),
                    "soft_mask_rel": f"{output_root_name}/{fusion_key}/{sample_id}.npy",
                }
            )
            f_decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")

            summary_by_subset[subset]["retained"] += 1
            soft_keep_ratios.append(soft_keep_ratio)
            target_keep_ratios.append(target_keep_ratio)
            global_positive_keep_ratios.append(global_positive_keep_ratio)
            global_failure_keep_ratios.append(global_failure_keep_ratio)

    stale_mask_count = 0
    if soft_mask_out_dir.exists():
        for npy_path in soft_mask_out_dir.glob("*.npy"):
            if npy_path.stem not in written_sample_ids:
                npy_path.unlink()
                stale_mask_count += 1

    samples_blob = {
        "schema": "ssf_family_beta_soft_label_v1",
        "train_dir": str(root.resolve()),
        "fusion_key": fusion_key,
        "output_root_name": output_root_name,
        "samples": output_samples,
        "input_summary": input_summary,
    }
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples_blob, f, ensure_ascii=False, indent=2)

    def _serialize_mask_reliability(mask_stats: dict[str, dict[str, float | int | str]]) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "family": str(mask_stats[key]["family"]),
                "num_present": int(mask_stats[key]["num_present"]),
                "num_correct": int(mask_stats[key]["num_correct"]),
                "acc": round(float(mask_stats[key]["acc"]), 6),
                "avg_keep_ratio": round(float(mask_stats[key]["avg_keep_ratio"]), 6),
                "evidence_efficiency": round(float(mask_stats[key]["evidence_efficiency"]), 6),
                "alpha": round(float(mask_stats[key]["alpha"]), 6),
            }
            for key in mask_keys
        }

    def _serialize_family_reliability(family_stats: dict[str, dict[str, float | int | str]]) -> dict[str, dict[str, Any]]:
        return {
            family_name: {
                "mask_keys": list(family_stats[family_name]["mask_keys"]),
                "num_mask_keys": int(family_stats[family_name]["num_mask_keys"]),
                "beta_reduce": "max",
                "family_evidence_efficiency": round(float(family_stats[family_name]["family_evidence_efficiency"]), 6),
                "beta": round(float(family_stats[family_name]["beta"]), 6),
            }
            for family_name in family_to_mask_keys
        }

    mask_reliability_summary = _serialize_mask_reliability(global_mask_stats)
    family_reliability_summary = _serialize_family_reliability(global_family_stats)
    subset_mask_reliability_summary = {
        subset: _serialize_mask_reliability(mask_stats)
        for subset, mask_stats in subset_mask_stats.items()
    }
    subset_family_reliability_summary = {
        subset: _serialize_family_reliability(family_stats)
        for subset, family_stats in subset_family_stats.items()
    }

    summary = {
        "schema": "ssf_family_beta_soft_label_v1",
        "train_dir": str(root.resolve()),
        "samples_json": str(Path(samples_json).resolve()),
        "fusion_key": fusion_key,
        "output_root_name": output_root_name,
        "epsilon": epsilon,
        "family_beta_reduce": "max",
        "mask_keys": mask_keys,
        "inference_keys": inference_keys,
        "family_to_mask_keys": family_to_mask_keys,
        "mask_reliability": mask_reliability_summary,
        "family_reliability": family_reliability_summary,
        "subset_mask_reliability": subset_mask_reliability_summary,
        "subset_family_reliability": subset_family_reliability_summary,
        "total_samples": len(samples),
        "retained_samples": len(output_samples),
        "dropped_samples": dropped_no_correct_count,
        "failed_samples": len(samples) - len(output_samples) - dropped_no_correct_count,
        "missing_inference_count": missing_inference_count,
        "dropped_no_correct_count": dropped_no_correct_count,
        "failure_reasons": dict(failure_reasons),
        "subset_stats": dict(summary_by_subset),
        "n_correct_distribution": {str(k): int(v) for k, v in sorted(n_correct_dist.items())},
        "n_wrong_distribution": {str(k): int(v) for k, v in sorted(n_wrong_dist.items())},
        "n_loaded_family_distribution": {str(k): int(v) for k, v in sorted(n_loaded_family_dist.items())},
        "n_active_family_distribution": {str(k): int(v) for k, v in sorted(n_active_family_dist.items())},
        "correct_method_combo_distribution": dict(correct_method_combo_dist),
        "wrong_method_combo_distribution": dict(wrong_method_combo_dist),
        "loaded_family_combo_distribution": dict(loaded_family_combo_dist),
        "active_family_combo_distribution": dict(active_family_combo_dist),
        "winning_method_distribution": dict(winning_method_dist),
        "winning_method_family_distribution": dict(winning_method_family_dist),
        "avg_soft_keep_ratio": round(float(np.mean(soft_keep_ratios)), 4) if soft_keep_ratios else None,
        "avg_target_keep_ratio": round(float(np.mean(target_keep_ratios)), 4) if target_keep_ratios else None,
        "avg_global_positive_keep_ratio": round(float(np.mean(global_positive_keep_ratios)), 4)
        if global_positive_keep_ratios
        else None,
        "avg_global_failure_keep_ratio": round(float(np.mean(global_failure_keep_ratios)), 4)
        if global_failure_keep_ratios
        else None,
        "stale_mask_removed_count": stale_mask_count,
        "outputs": {
            "soft_mask_dir": str(soft_mask_out_dir),
            "records_jsonl": str(records_path),
            "samples_json": str(samples_path),
            "decisions_jsonl": str(decisions_path),
            "summary_json": str(summary_path),
            "hard_failure_pool_jsonl": str(hard_failure_pool_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTrain Success-Failure Family-Beta Soft Label build done:")
    print(f"  total samples               : {len(samples)}")
    print(f"  retained samples            : {len(output_samples)}")
    print(f"  dropped no-correct          : {dropped_no_correct_count}")
    print(f"  failed samples              : {len(samples) - len(output_samples) - dropped_no_correct_count}")
    if soft_keep_ratios:
        print(f"  avg soft_keep_ratio         : {np.mean(soft_keep_ratios):.4f}")
    if global_positive_keep_ratios:
        print(f"  avg global_positive_ratio   : {np.mean(global_positive_keep_ratios):.4f}")
    if global_failure_keep_ratios:
        print(f"  avg global_failure_ratio    : {np.mean(global_failure_keep_ratios):.4f}")
    if target_keep_ratios:
        print(f"  avg target_keep_ratio       : {np.mean(target_keep_ratios):.4f}")
    print("  family_beta_reduce          : max")
    print(f"  stale masks removed         : {stale_mask_count}")
    print(f"  soft masks                  : {soft_mask_out_dir}")
    print(f"  records                     : {records_path}")
    print(f"  samples                     : {samples_path}")
    print(f"  decisions                   : {decisions_path}")
    print(f"  hard failure pool           : {hard_failure_pool_path}")
    print(f"  summary                     : {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build SSF family-beta soft labels from existing train_inference results."
    )
    p.add_argument("--train_dir", default="./train_data_uid_qwenv2")
    p.add_argument(
        "--samples_json",
        default="./train_data_uid_qwenv2/sample_list.json",
        help="sample_list.json 或其他兼容 manifest",
    )
    p.add_argument("--mask_keys", nargs="+", default=DEFAULT_MASK_KEYS, help="参与融合的 mask key 列表")
    p.add_argument("--run_prefix", default="train_", help="当未显式提供 inference_keys 时，按 run_prefix + mask_key 推导")
    p.add_argument(
        "--inference_keys",
        nargs="+",
        default=None,
        help="可选：显式指定 train_inference/*.jsonl 的 run_name 列表",
    )
    p.add_argument(
        "--fusion_key",
        default=None,
        help="输出 fusion key；默认自动使用 ssf_family_beta_soft_label",
    )
    p.add_argument("--output_root_name", default="soft_masks")
    p.add_argument("--epsilon", type=float, default=1e-6, help="evidence efficiency 数值稳定项")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_build_train_soft_labels_success_failure_family_beta(
        train_dir=args.train_dir,
        samples_json=args.samples_json,
        mask_keys=args.mask_keys,
        run_prefix=args.run_prefix,
        inference_keys=args.inference_keys,
        fusion_key=args.fusion_key,
        output_root_name=args.output_root_name,
        epsilon=args.epsilon,
    )
