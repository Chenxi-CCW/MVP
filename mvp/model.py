from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_ffn_dim(hidden_dim: int, ffn_dim: int | None) -> int:
    return int(hidden_dim * 4 if ffn_dim is None else ffn_dim)


def extract_model_config(source: dict | None = None) -> dict[str, int | float | str]:
    source = source or {}
    hidden_dim = int(source.get("hidden_dim", 512))
    return {
        "visual_input_dim": int(source.get("visual_input_dim", 1024)),
        "text_input_dim": int(source.get("text_input_dim", 4096)),
        "hidden_dim": hidden_dim,
        "num_layers": int(source.get("num_layers", 2)),
        "num_heads": int(source.get("num_heads", 8)),
        "ffn_dim": _default_ffn_dim(hidden_dim, source.get("ffn_dim")),
        "dropout": float(source.get("dropout", 0.1)),
        "num_patches": int(source.get("num_patches", 576)),
        "block_type": str(source.get("block_type", "self_cross_swiglu")),
    }


def model_config_from_checkpoint(ckpt: dict) -> dict[str, int | float | str]:
    if "model_config" in ckpt:
        return extract_model_config(ckpt["model_config"])
    if "args" in ckpt:
        return extract_model_config(ckpt["args"])
    return extract_model_config()


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_dim, ffn_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.dropout(x)
        return self.down_proj(x)


class MVPPrunerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_ctx, _ = self.visual_self_attn(
            query=visual_tokens,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=False,
        )
        visual_tokens = self.norm_self(visual_tokens + self.dropout(visual_ctx))

        key_padding_mask = None
        if question_attention_mask is not None:
            key_padding_mask = ~question_attention_mask.bool()

        text_ctx, _ = self.text_cross_attn(
            query=visual_tokens,
            key=question_tokens,
            value=question_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        visual_tokens = self.norm_cross(visual_tokens + self.dropout(text_ctx))
        visual_tokens = self.norm_ffn(visual_tokens + self.dropout(self.ffn(visual_tokens)))
        return visual_tokens


class MVPPruner(nn.Module):
    def __init__(
        self,
        visual_input_dim: int = 1024,
        text_input_dim: int = 4096,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int | None = 2048,
        dropout: float = 0.1,
        num_patches: int = 576,
        block_type: str = "self_cross_swiglu",
    ) -> None:
        super().__init__()
        if block_type != "self_cross_swiglu":
            raise ValueError(f"Unsupported block_type: {block_type}")

        ffn_dim = _default_ffn_dim(hidden_dim, ffn_dim)
        self.num_patches = int(num_patches)
        self.model_config = {
            "visual_input_dim": int(visual_input_dim),
            "text_input_dim": int(text_input_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "ffn_dim": int(ffn_dim),
            "dropout": float(dropout),
            "num_patches": int(num_patches),
            "block_type": str(block_type),
        }

        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(visual_input_dim),
            nn.Linear(visual_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_adapter = nn.Sequential(
            nn.LayerNorm(text_input_dim),
            nn.Linear(text_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.ModuleList(
            [
                MVPPrunerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.scoring_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be 3D, got {tuple(visual_tokens.shape)}")
        if question_tokens.ndim != 3:
            raise ValueError(f"question_tokens must be 3D, got {tuple(question_tokens.shape)}")
        if visual_tokens.size(1) != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patch tokens, got {visual_tokens.size(1)}"
            )

        visual_hidden = self.visual_adapter(visual_tokens)
        question_hidden = self.text_adapter(question_tokens)
        for layer in self.encoder:
            visual_hidden = layer(
                visual_tokens=visual_hidden,
                question_tokens=question_hidden,
                question_attention_mask=question_attention_mask,
            )

        return self.scoring_head(visual_hidden).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(
                visual_tokens=visual_tokens,
                question_tokens=question_tokens,
                question_attention_mask=question_attention_mask,
            )
        )

    @classmethod
    def from_config(cls, config: dict | None = None) -> "MVPPruner":
        return cls(**extract_model_config(config))
