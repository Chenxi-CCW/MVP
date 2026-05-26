"""
extractors/scope_extractor.py

SCOPE (NeurIPS 2025) mask 计算器，严格对齐官方实现。

官方：scope/clip_encoder.py  CLIPVisionTower_SCOPE.forward + SCOPE()

gains[j] = sum_i max(0, simi[i,j] - cur_max[i])
    - 掩列 j（j 已选）：simi.masked_fill(~unsel.unsqueeze(0), 0)  → (N,N), unsqueeze(0)→(1,N) 掩列
    - 减 cur_max[i]：cur_max.unsqueeze(1) → (N,1)
    - sum(dim=0)：对 i 维求和
cur_max 更新：simi[best, :]（取 best 行）

推理：纯 patch tokens，不含 CLS。
"""

import numpy as np
import torch
from typing import Dict, List

N_PATCHES = 576
_ALPHA    = 1.0


def _scope_select(
    patch_feats: torch.Tensor,  # (576, D) float32
    cls_attn:    torch.Tensor,  # (576,) float32，heads sum
    k: int,
) -> np.ndarray:
    """SCOPE 贪心，单样本，严格对齐官方 SCOPE()。"""
    N      = patch_feats.shape[0]
    device = patch_feats.device
    dtype  = patch_feats.dtype

    norm    = patch_feats / patch_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    simi    = torch.mm(norm, norm.T)          # (N, N)  simi[i,j]
    cls_pow = cls_attn ** _ALPHA              # (N,)

    selected = torch.zeros(N, dtype=torch.bool, device=device)
    sel_idx  = torch.empty(k, dtype=torch.long, device=device)
    cur_max  = torch.zeros(N, dtype=dtype, device=device)

    for i in range(k):
        unsel = ~selected                     # (N,) True=未选

        # 掩列 j（j 已选则置 0），对应官方 masked_fill(~unsel.unsqueeze(1), 0) B=1 展开
        # ~unsel.unsqueeze(0): (1,N) 广播到 (N,N)，掩列
        masked_simi = simi.masked_fill(~unsel.unsqueeze(0), 0.0)   # (N, N)

        # gains[j] = sum_i max(0, simi[i,j] - cur_max[i])
        # cur_max.unsqueeze(1): (N,1)，每行减 cur_max[i]
        # sum(dim=0)：对 i 维求和
        gains = (masked_simi - cur_max.unsqueeze(1)).clamp(min=0).sum(dim=0)  # (N,)

        gains = gains * cls_pow               # 乘 CLS attention 权重
        gains[selected] = float("-inf")       # 已选不参与

        best           = gains.argmax()
        selected[best] = True
        sel_idx[i]     = best
        cur_max        = torch.maximum(cur_max, simi[best, :])  # 取 best 行

    return sel_idx.cpu().numpy()


class SCOPEExtractor:
    """从 ViT 倒数第二层结果计算 SCOPE patch mask，无需重跑 ViT。"""

    def __init__(self, token_counts: List[int] = (64, 128)):
        self.token_counts = list(token_counts)

    def compute_masks(
        self,
        hidden_pen: np.ndarray,  # (577, D) float16
        attn_pen:   np.ndarray,  # (heads, 577, 577) float32
    ) -> Dict[int, np.ndarray]:
        """
        Returns:
            {n_tokens: (576,) bool}，patch-only mask，True=保留。
        """
        # 官方：attn_weights[:, :, 0, 1:].sum(dim=1)，heads sum → (576,)
        cls_attn    = torch.from_numpy(attn_pen[:, 0, 1:]).float().sum(dim=0)

        # 官方：hidden_states[:, 1:]，去 CLS → (576, D)
        patch_feats = torch.from_numpy(hidden_pen[1:]).float()

        masks: Dict[int, np.ndarray] = {}
        for n in self.token_counts:
            k    = min(n, N_PATCHES)
            sel  = _scope_select(patch_feats, cls_attn, k=k)
            mask = np.zeros(N_PATCHES, dtype=bool)
            mask[sel] = True
            masks[n]  = mask
        return masks