from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer


def _ensure_llava_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    candidates = []
    if os.environ.get("LLAVA_REPO"):
        candidates.append(Path(os.environ["LLAVA_REPO"]).expanduser().resolve())
    candidates.append(project_root / "LLaVA")
    for llava_root in candidates:
        if str(llava_root) not in sys.path:
            sys.path.insert(0, str(llava_root))


def _resolve_llava_model_cls(llava_path: str):
    _ensure_llava_import_path()
    config = AutoConfig.from_pretrained(llava_path)
    architectures = [str(x) for x in getattr(config, "architectures", [])]
    arch = architectures[0] if architectures else ""

    if arch == "LlavaLlamaForCausalLM":
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

        return LlavaLlamaForCausalLM
    if arch == "LlavaMistralForCausalLM":
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

        return LlavaMistralForCausalLM
    if arch == "LlavaMptForCausalLM":
        from llava.model.language_model.llava_mpt import LlavaMptForCausalLM

        return LlavaMptForCausalLM

    raise ValueError(
        f"Unsupported LLaVA architecture: model_type={config.model_type!r}, "
        f"architectures={architectures!r}"
    )


class LlavaQuestionEmbedder(nn.Module):
    def __init__(
        self,
        llava_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        _ensure_llava_import_path()
        import llava.model  # noqa: F401

        self.device_name = device
        self.model_dtype = self._parse_dtype(dtype, device)
        model_cls = _resolve_llava_model_cls(llava_path)

        self.tokenizer = AutoTokenizer.from_pretrained(llava_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model = model_cls.from_pretrained(
            llava_path,
            low_cpu_mem_usage=True,
            dtype=self.model_dtype,
        )
        self.model.eval()
        self.model.to(device)
        self.model.requires_grad_(False)

        self.hidden_size = int(self.model.config.hidden_size)
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    @staticmethod
    def _parse_dtype(dtype: str, device: str) -> torch.dtype:
        if device == "cpu":
            return torch.float32
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.model.device)
        return self.model.get_model().embed_tokens(input_ids)
