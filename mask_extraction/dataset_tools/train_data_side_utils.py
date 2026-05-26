from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def get_default_image_token() -> str:
    """
    获取 LLaVA 默认图像 token。

    正常项目里会从 LLaVA.llava.constants 读取。
    如果当前环境暂时无法 import LLaVA，则 fallback 到 "<image>"。
    """
    try:
        from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN
        return DEFAULT_IMAGE_TOKEN
    except Exception:
        return "<image>"


def clean_question_text(question: str) -> str:
    """
    清洗 adapter 训练用的 question 原文。

    兼容输入：
    1. "What color is the cat?"
    2. "<image>\nWhat color is the cat?"     ← <image> 开头
    3. "What color is the cat?\n<image>"     ← <image> 结尾
    """
    text = str(question)
    default_image_token = get_default_image_token()

    # 删除所有 <image> token，不依赖其位置
    text = text.replace(default_image_token, "")

    # 去掉首尾格式空白（含 \n、\r\n、空格、tab）
    text = text.strip()

    if not text:
        raise ValueError(f"Empty question after cleaning. Raw question: {question!r}")

    return text


def encode_question_tokens(
    tokenizer,
    question: str,
    max_text_len: int | None = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    抽取 adapter 训练用 question token。

    输入：
        tokenizer:
            LLaVA / Vicuna / LLaMA 对应 tokenizer

        question:
            原始问题文本，建议直接来自 sample["question"]

        max_text_len:
            最大 question token 长度。
            如果为 None，则不截断。

    输出：
        input_ids:
            shape: (L_q,)
            dtype: np.int32

        attention_mask:
            shape: (L_q,)
            dtype: np.int8

    重要：
        这里保存的是 question input_ids，不是 4096 维 embedding。

        后续 adapter 训练时再用：
            model.get_model().embed_tokens(input_ids)

        得到：
            question_embeds: (L_q, 4096)
    """
    question_text = clean_question_text(question)

    enc = tokenizer(
        question_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )

    # 兼容 HuggingFace tokenizer 返回 BatchEncoding 或 dict 的情况
    if hasattr(enc, "input_ids"):
        input_ids_list = enc.input_ids
    else:
        input_ids_list = enc["input_ids"]

    input_ids = np.asarray(input_ids_list, dtype=np.int32)

    if input_ids.ndim != 1:
        raise ValueError(
            f"Expected 1D input_ids, got shape={input_ids.shape}, "
            f"question={question_text!r}"
        )

    if len(input_ids) == 0:
        raise ValueError(
            f"Tokenizer returned empty input_ids for question: {question_text!r}"
        )

    if max_text_len is not None and len(input_ids) > max_text_len:
        input_ids = input_ids[:max_text_len]

    attention_mask = np.ones_like(input_ids, dtype=np.int8)
    return input_ids, attention_mask


class QuestionTokenStore:
    """
    保存 adapter 训练用 question tokens。

    目录结构：
        output_dir/
            question_tokens/
                {sample_id}.npz

    每个 npz 内部：
        input_ids       : int32, shape (L_q,)
        attention_mask  : int8,  shape (L_q,)
    """

    def __init__(self, output_dir: str):
        self.root = Path(output_dir) / "question_tokens"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sample_id: str) -> Path:
        return self.root / f"{sample_id}.npz"

    def is_done(self, sample_id: str) -> bool:
        return self.path(sample_id).exists()

    def save(
        self,
        sample_id: str,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if input_ids.ndim != 1:
            raise ValueError(
                f"input_ids must be 1D, got shape={input_ids.shape}, "
                f"sample_id={sample_id}"
            )

        if attention_mask.ndim != 1:
            raise ValueError(
                f"attention_mask must be 1D, got shape={attention_mask.shape}, "
                f"sample_id={sample_id}"
            )

        if len(input_ids) != len(attention_mask):
            raise ValueError(
                f"input_ids and attention_mask length mismatch: "
                f"{len(input_ids)} vs {len(attention_mask)}, "
                f"sample_id={sample_id}"
            )

        np.savez_compressed(
            self.path(sample_id),
            input_ids=input_ids.astype(np.int32, copy=False),
            attention_mask=attention_mask.astype(np.int8, copy=False),
        )


class QuestionTokenBuilder:
    """
    adapter 训练用 question token builder。

    与旧版区别：
    - 不构造完整 LLaVA prompt
    - 不从 <image> 和 ASSISTANT 之间切 span
    - 不保留 leading "\n"
    - 不保存 USER / ASSISTANT / system prompt
    - 不保存 BOS token
    - 直接 tokenize 纯 question
    """

    def __init__(self, tokenizer, max_text_len: int = 256):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def encode(self, question: str) -> tuple[np.ndarray, np.ndarray]:
        return encode_question_tokens(
            tokenizer=self.tokenizer,
            question=question,
            max_text_len=self.max_text_len,
        )


# 兼容旧调用方命名
TextTokenStore = QuestionTokenStore


def load_question_token_npz(path: str | Path) -> dict[str, np.ndarray]:
    """
    读取已经保存的 question token npz。

    返回：
        {
            "input_ids": np.ndarray[int32], shape (L_q,)
            "attention_mask": np.ndarray[int8], shape (L_q,)
        }
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Question token file not found: {path}")

    obj = np.load(path)

    if "input_ids" not in obj:
        raise KeyError(f"'input_ids' not found in {path}")

    if "attention_mask" not in obj:
        raise KeyError(f"'attention_mask' not found in {path}")

    input_ids = obj["input_ids"].astype(np.int32, copy=False)
    attention_mask = obj["attention_mask"].astype(np.int8, copy=False)

    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape={input_ids.shape}, path={path}")

    if attention_mask.ndim != 1:
        raise ValueError(
            f"attention_mask must be 1D, got shape={attention_mask.shape}, path={path}"
        )

    if len(input_ids) != len(attention_mask):
        raise ValueError(
            f"Length mismatch in {path}: "
            f"input_ids={len(input_ids)}, attention_mask={len(attention_mask)}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def load_final_samples_json(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    读取前置阶段输出的最终样本 JSON。

    支持格式：
      - {"samples": [...], ...}
      - 直接是 list[sample]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"samples_json not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    meta: dict[str, Any] = {}
    if isinstance(obj, dict):
        raw_samples = obj.get("samples")
        meta = {k: v for k, v in obj.items() if k != "samples"}
    elif isinstance(obj, list):
        raw_samples = obj
        meta = {"schema": "sample_list"}
    else:
        raise ValueError(f"Unsupported samples_json format: {p}")

    if not isinstance(raw_samples, list):
        raise ValueError(f"samples_json must contain a sample list: {p}")

    required_keys = {"uid", "sample_id", "orig_id", "image_path", "question", "answer", "subset"}
    samples: list[dict[str, Any]] = []
    seen_uids: set[str] = set()

    for idx, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            raise ValueError(f"Sample #{idx} is not a dict in {p}")
        missing = sorted(required_keys - set(item.keys()))
        if missing:
            raise ValueError(f"Sample #{idx} missing keys {missing} in {p}")

        sample = dict(item)
        sample["uid"] = str(sample["uid"])
        sample["sample_id"] = str(sample["sample_id"])
        sample["orig_id"] = str(sample["orig_id"])
        sample["image_path"] = str(sample["image_path"])
        sample["question"] = str(sample["question"])
        sample["answer"] = str(sample["answer"])
        sample["subset"] = str(sample["subset"])
        if "image_rel" in sample and sample["image_rel"] is not None:
            sample["image_rel"] = str(sample["image_rel"])

        if sample["uid"] in seen_uids:
            continue
        seen_uids.add(sample["uid"])
        samples.append(sample)

    if not samples:
        raise RuntimeError(f"No samples loaded from {p}")

    subset_dist = dict(Counter(str(s["subset"]) for s in samples))
    summary = {
        "samples_json": str(p),
        "total_loaded": len(samples),
        "subset_distribution": subset_dist,
        "manifest_meta": meta,
    }
    return samples, summary


def write_sample_list(sample_list_path: Path, samples: list[dict[str, Any]], merge_existing: bool = True) -> int:
    merged_by_uid: dict[str, dict[str, Any]] = {}

    if merge_existing and sample_list_path.exists():
        with open(sample_list_path, "r", encoding="utf-8") as f:
            try:
                old_samples = json.load(f)
            except Exception:
                old_samples = []
        if isinstance(old_samples, list):
            for s in old_samples:
                if isinstance(s, dict) and s.get("uid"):
                    merged_by_uid[str(s["uid"])] = s

    for s in samples:
        merged_by_uid[str(s["uid"])] = s

    merged = list(merged_by_uid.values())
    sample_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_list_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return len(merged)


def split_samples(samples: list[dict[str, Any]], n_parts: int) -> list[list[dict[str, Any]]]:
    if n_parts <= 0:
        return []
    chunk_size = math.ceil(len(samples) / n_parts)
    return [samples[i * chunk_size:(i + 1) * chunk_size] for i in range(n_parts)]


def parse_gpu_ids(gpus: str) -> list[int]:
    gpu_ids = [int(x.strip()) for x in str(gpus).split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided.")
    return gpu_ids
