"""
extractors/vispruner_extractor.py

VisPruner (ICCV 2025) mask 计算器，严格对齐官方实现（数值验证通过）。

官方路径：
    llava/model/multimodal_encoder/clip_encoder.py  CLIPVisionTower.forward
        feature_select (select_feature='patch', output_attentions=True):
            image_features   = hidden_states[select_layer][:, 1:]       → (B, 576, D)
            image_attentions = attentions[select_layer][:, :, 0, 1:]    → (B, heads, 576)
    llava/model/llava_arch.py  encode_images()

Stage 1 — 重要性：
    cls_attn = image_attentions.mean(dim=1)             → (B, 576)  heads mean
    token_indices = cls_attn.argsort(descending=True)   → (B, 576)
    important_indices = token_indices[:, :k_imp]
    residual_indices  = token_indices[:, k_imp:]        ← 保持降序排列的顺序！

Stage 2 — 多样性（iterative pair-wise pruning）：
    residual_tokens = image_normalized[residual_indices]   (R, D)
    a = residual_tokens[::2]    (ceil(R/2), D)  偶数位
    b = residual_tokens[1::2]   (floor(R/2), D) 奇数位
    scores = (a @ b^T).max(dim=-1).values       (ceil(R/2),)
        每个偶数 token 与所有奇数 token 的最大余弦相似度
    distinct_indices = scores.argsort(descending=True)[r:]
        去掉前 r 个（最相似），保留 (ceil(R/2)-r) 个偶数侧
    residual_indices = cat(偶数侧筛选后的原始索引, 奇数侧全部原始索引)
    r = min(8, R - k_div)

推理：纯 patch tokens，不含 CLS。
"""

import numpy as np
import torch
from typing import Dict, List

N_PATCHES  = 576
_IMP_RATIO = 0.5  # 官方默认


def _vispruner_select(
    patch_feats: torch.Tensor,  # (576, D) float32，已去 CLS
    cls_attn:    torch.Tensor,  # (576,) float32，heads mean
    k: int,
) -> np.ndarray:
    """
    VisPruner 两阶段选取，单样本，数值对齐官方 encode_images()（验证通过）。

    关键：residual_indices 初始顺序 = cls_attn 降序排列的后半段，
         与官方 token_indices[k_imp:] 完全一致（非原始 patch 顺序）。

    Returns:
        (k,) int64，0-indexed patch 索引（对应 576 patch 空间）。
    """
    device = patch_feats.device
    k_imp  = max(1, int(k * _IMP_RATIO))
    k_div  = k - k_imp

    # Stage 1：官方 token_indices = argsort(descending)，前 k_imp 为重要，后面为剩余
    token_indices    = cls_attn.argsort(descending=True)    # (576,)
    important_idx    = token_indices[:k_imp]                 # (k_imp,) 原始 patch 索引
    residual_indices = token_indices[k_imp:]                 # (576-k_imp,) 降序剩余，保持此顺序！

    # 归一化特征（Stage 2 用）
    image_normalized = patch_feats / patch_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Stage 2：iterative pair-wise pruning
    while k_div > 0:
        R = len(residual_indices)
        r = min(8, R - k_div)
        if r <= 0:
            break

        # 按 residual_indices 取归一化特征（保持 residual 顺序）
        residual_tokens  = image_normalized[residual_indices]     # (R, D)

        a = residual_tokens[0::2]                                 # (ceil(R/2), D) 偶数位
        b = residual_tokens[1::2]                                 # (floor(R/2), D) 奇数位

        # 每个偶数 token 与所有奇数 token 的最大余弦相似度
        scores = (a @ b.T).max(dim=-1).values                    # (ceil(R/2),)

        n_even           = len(a)                                 # ceil(R/2)
        # 降序取 [r:]：去掉最相似的前 r 个，保留后面相似度较低的
        distinct_indices = scores.argsort(descending=True)[r:]    # (n_even - r,)

        residual_indices = torch.cat([
            residual_indices[0::2][distinct_indices],             # 筛选后偶数侧原始 patch 索引
            residual_indices[1::2],                               # 全部奇数侧原始 patch 索引
        ])

    # 合并
    if k_div > 0:
        selected = torch.cat([important_idx, residual_indices[:k_div]])
    else:
        selected = important_idx

    return selected.cpu().numpy()


class VisPrunerExtractor:
    """从 ViT 倒数第二层结果计算 VisPruner patch mask，无需重跑 ViT。"""

    def __init__(self, token_counts: List[int] = (64, 128)):
        self.token_counts = list(token_counts)

    def compute_masks(
        self,
        hidden_pen: np.ndarray,  # (577, D) float16
        attn_pen:   np.ndarray,  # (heads, 577, 577) float32
    ) -> Dict[int, np.ndarray]:
        """
        Args:
            hidden_pen: ViT 倒数第二层 hidden，含 CLS，(577, D)。
            attn_pen  : ViT 倒数第二层 attention，(heads, 577, 577)。

        Returns:
            {n_tokens: (576,) bool}，patch-only mask，True=保留。
        """
        # 官方 feature_select: attentions[select_layer][:, :, 0, 1:] → (B, heads, 576)
        # encode_images: .mean(dim=1) → (B, 576)
        # 单样本展开：attn_pen[:, 0, 1:] → (heads, 576)，mean(dim=0) → (576,)
        cls_attn    = torch.from_numpy(attn_pen[:, 0, 1:]).float().mean(dim=0)  # (576,)

        # 官方 feature_select: hidden_states[select_layer][:, 1:] → (B, 576, D)
        # 单样本：hidden_pen[1:] → (576, D)
        patch_feats = torch.from_numpy(hidden_pen[1:]).float()                  # (576, D)

        masks: Dict[int, np.ndarray] = {}
        for n in self.token_counts:
            k    = min(n, N_PATCHES)
            sel  = _vispruner_select(patch_feats, cls_attn, k=k)
            mask = np.zeros(N_PATCHES, dtype=bool)
            mask[sel] = True
            masks[n]  = mask
        return masks