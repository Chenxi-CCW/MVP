"""
extractors/GRASP.py

GRASP (Question-adaptive Object-centric Token Allocation) mask 计算器。

该 extractor 完全独立加载 Qwen3-VL 和 SAM3，不借用外部模型封装。

流程：
    1. Qwen3-VL → targets（text prompt 列表）
    2. 每个 target 独立 set_image + set_text_prompt → pixel mask
    3. 多 target masks logical_or 合并
    4. transform_mask_like_clip：longest-edge resize → 对称 pad → 336×336
    5. mask_to_token_indices：reshape → mean → >0 → (576,) bool

输出：(576,) bool，True=保留该 patch。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


# _TARGET_EXTRACTION_PROMPT = """You are a visual grounding assistant for SAM3, a text-prompted segmentation model.
# Given an image and a question, return the smallest set of visible targets that must be preserved so the question can still be answered.

# Question: "{question}"

# ## Core rule
# Return the minimal set of segmentable visual evidence. NOT the answer, NOT abstract words. When uncertain, include a small number of additional relevant objects (up to 4 total) rather than missing a necessary one.

# ## Output format
# - 1 to 4 noun phrases, comma-separated, single line
# - No explanation, no prefix, no full sentence

# ## Every target must be
# - visually grounded (you actually see it in the image)
# - a concrete, localizable object or region
# - a short lowercase noun phrase, each target <= 3 words

# ## Question-type guide

# 1. Attribute / color / count / existence -> output the OBJECT
# "What color is the cat?"          -> cat
# "How many red cars are parked?"   -> red car

# 2. Relation / spatial -> include ALL named objects
# "Is the cup on the table?"        -> cup, table

# 3. Open-ended "what's on / in / near / around X" -> output X AND the items you ACTUALLY SEE there (use your vision)
# Image: apples & bananas on a table.
# "What's on the table?"            -> table, apple, banana
# Image: books on a shelf.
# "What's on the shelf?"            -> shelf, book

# 4. Pure number / code / text (no visible carrier) -> `word`
# "What is the result of the code?" -> word

# 5. Text / OCR / reading / sign content (with visible carrier) -> `word` always, paired with the carrier region
# "What does the sign say?"         -> sign, word
# "Who wrote the book?"             -> book cover, word

# 6. Scene / indoor-outdoor / weather / time-of-day -> characteristic elements you actually see
# Image: bedroom.  "Indoor or outdoor?" -> wall, bed
# Image: beach.    "Is it sunny?"       -> sky, sand

# 7. Chart / diagram / science figure (ScienceQA, MMVet)
# "What does the graph show?"       -> chart, word
# "What does the diagram show?"     -> diagram, word

# 8. Whole-image / "what is this image about"
# -> 2-3 most salient visible objects

# ## Do NOT output
# - background, scene, environment, activity, event
# - answer words, yes/no, verbs alone
# - abstract or knowledge concepts not directly visible
# - `text`, `text region` (use `word` instead)

# ## Output
# Return ONLY the targets, comma-separated.
# Targets:"""

_TARGET_EXTRACTION_PROMPT = """You are a visual grounding assistant for SAM3, a text-prompted segmentation model.
Given an image and a question, return the smallest set of visible targets that must be preserved so the question can still be answered.

Question: "{question}"

## Core rule
Return the minimal set of segmentable visual evidence. NOT the answer, NOT abstract words. When uncertain, include a small number of additional relevant objects (up to 4 total) rather than missing a necessary one.

## Output format
- 1 to 4 noun phrases, comma-separated, single line
- No explanation, no prefix, no full sentence

## Every target must be
- visually grounded (you actually see it in the image)
- a concrete, localizable object or region
- a short lowercase noun phrase, each target <= 3 words

## Question-type guide

1. Attribute / color / count / existence -> output the OBJECT
"What color is the cat?"          -> cat
"How many red cars are parked?"   -> red car

2. Relation / spatial -> include ALL named objects
"Is the cup on the table?"        -> cup, table

3. Open-ended "what's on / in / near / around X" -> output X AND the items you ACTUALLY SEE there (use your vision)
Image: apples & bananas on a table.
"What's on the table?"            -> table, apple, banana
Image: books on a shelf.
"What's on the shelf?"            -> shelf, book

4. Pure number / code / text (no visible carrier) -> `word`
"What is the result of the code?" -> word

5. Text / OCR / reading / sign content (with visible carrier) -> `word` always
"What does the sign say?"         -> word
"Who wrote the book?"             -> word

6. Scene / indoor-outdoor / weather / time-of-day -> characteristic elements you actually see
Image: bedroom.  "Indoor or outdoor?" -> wall, bed
Image: beach.    "Is it sunny?"       -> sky, sand

7. Chart / diagram / science figure (ScienceQA, MMVet)
"What does the graph show?"       -> chart, word
"What does the diagram show?"     -> diagram, word

8. Whole-image / "what is this image about"
-> 2-3 most salient visible objects

## Do NOT output
- background, scene, environment, activity, event
- answer words, yes/no, verbs alone
- abstract or knowledge concepts not directly visible
- `text`, `text region` (use `word` instead)

## Output
Return ONLY the targets, comma-separated.
Targets:"""


# ── CLIP 对齐的 mask 变换（对齐 token_pruning.transform_mask_like_clip）────────

def _transform_mask_like_clip(
    mask: np.ndarray,   # (H, W) bool，原始图像空间
    orig_w: int,
    orig_h: int,
    clip_size: int = 336,
) -> np.ndarray:
    """
    对 mask 施加与 CLIPImageProcessor 完全相同的预处理：
        1. Longest-edge resize 到 clip_size，保持宽高比
        2. 短边对称 pad 到 clip_size × clip_size，pad 区域填 0

    Returns:
        (336, 336) bool，与 ViT patch grid 对齐。
    """
    scale = clip_size / max(orig_h, orig_w)
    new_h = round(orig_h * scale)
    new_w = round(orig_w * scale)

    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_pil.resize((new_w, new_h), Image.NEAREST)

    pad_top  = (clip_size - new_h) // 2
    pad_left = (clip_size - new_w) // 2

    canvas = Image.new("L", (clip_size, clip_size), 0)
    canvas.paste(mask_resized, (pad_left, pad_top))

    return np.array(canvas) > 127   # (336, 336) bool


# ── Mask → (576,) patch bool（对齐 token_pruning.mask_to_token_indices）────────

def _mask_to_patch_mask(
    mask_336: np.ndarray,   # (336, 336) bool
    grid_h: int = 24,
    grid_w: int = 24,
) -> np.ndarray:
    """
    将 336×336 的像素 mask 下采样到 24×24 patch grid。

    使用 mean pooling（不是 NEAREST resize）：
        reshape → (grid_h, patch_h, grid_w, patch_w) → mean(axis=(1,3))
    任意像素覆盖（coverage > 0）即保留该 patch（any() 语义）。

    Returns:
        (576,) bool。
    """
    mask_float = mask_336.astype(np.float32)

    patch_h = mask_336.shape[0] // grid_h   # 336 / 24 = 14
    patch_w = mask_336.shape[1] // grid_w

    tiled          = mask_float.reshape(grid_h, patch_h, grid_w, patch_w)
    patch_coverage = tiled.mean(axis=(1, 3))    # (24, 24)
    patch_flags    = (patch_coverage > 0).flatten()   # (576,) bool

    return patch_flags


# ── GRASPExtractor ─────────────────────────────────────────────────────────────

class GRASPExtractor:
    """
    GRASP mask 计算器：Qwen3-VL text grounding → SAM3 text prompt → patch mask。

    独立流程：
        - text prompt（非 bbox prompt）
        - 每个 target 独立 set_image + set_text_prompt
        - masks logical_or 合并
        - any() pooling（mean > 0）下采样到 24×24
        - 空 mask 直接 continue，不做面积过滤
    """

    def __init__(
        self,
        qwen_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        qwen_device: str = "cuda:1",
        sam3_device: str = "cuda:0",
        confidence_threshold: float = 0.5,
        clip_size: int = 336,
        grid_h: int = 24,
        grid_w: int = 24,
    ):
        """
        Args:
            qwen_path:    Qwen3-VL 权重路径。
            qwen_device:  Qwen3-VL 所在设备。
            sam3_device:  SAM3 所在设备。
            confidence_threshold: SAM3 text prompt 置信度阈值。
            clip_size:    CLIP 输入分辨率（336 for LLaVA-1.5）。
            grid_h/w:     Patch grid 大小（24×24 for ViT-L/14@336）。
        """
        self.clip_size = clip_size
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.qwen_device = torch.device(qwen_device)
        self.sam3_device = torch.device(sam3_device)

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        sam3_repo = os.environ.get("SAM3_REPO")
        if sam3_repo:
            sam3_path = Path(sam3_repo).expanduser().resolve()
            if str(sam3_path) not in sys.path:
                sys.path.insert(0, str(sam3_path))
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception:
            from test.sam3.sam3.model_builder import build_sam3_image_model
            from test.sam3.sam3.model.sam3_image_processor import Sam3Processor

        print(f"Loading Qwen3-VL from {qwen_path} on {qwen_device}...")
        try:
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": qwen_device},
            )
        except Exception:
            print("  Warning: flash_attention_2 not available, using default attention")
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_path,
                torch_dtype=torch.bfloat16,
                device_map={"": qwen_device},
            )
        self.qwen.eval()
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_path)
        print(f"✓ Qwen3-VL loaded on {qwen_device}")

        print(f"Loading SAM3 on {sam3_device}...")
        self.sam3_model = build_sam3_image_model(device=sam3_device)
        # 某些 SAM3 builder 只处理 "cuda"、不处理 "cuda:0/1"；
        # 这里再显式对齐一次，避免输入已经在 GPU、权重仍停在 CPU。
        self.sam3_model = self.sam3_model.to(self.sam3_device)
        self.sam3_processor = Sam3Processor(
            self.sam3_model,
            confidence_threshold=confidence_threshold,
            device=sam3_device,
        )
        print(f"✓ SAM3 loaded on {sam3_device}")

    def extract_targets(self, image: Image.Image, question: str) -> List[str]:
        """Extract visual segmentation targets using Qwen3-VL."""
        prompt = _TARGET_EXTRACTION_PROMPT.format(question=question)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.qwen_device)

        with torch.no_grad():
            generated_ids = self.qwen.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        targets = []
        for t in output_text.split(","):
            t = t.strip().lower()
            if t and len(t) < 50 and not t.startswith(
                ("output:", "example", "note", "question:")
            ):
                targets.append(t)
        return targets[:5] if targets else ["background"]

    def compute_mask(
        self,
        image: Image.Image,
        question: str,
    ) -> Tuple[np.ndarray, List[str], int, float]:
        """
        计算单张图的 GRASP patch mask。

        Args:
            image:    原始 PIL 图像（RGB）。
            question: 问题文本，传给 Qwen3-VL 提取 targets。

        Returns:
            mask     : (576,) bool，True=保留该 patch。
            targets  : Qwen3-VL 提取的 target 列表。
            n_valid  : 实际产生有效分割结果的 target 数。
            coverage : patch mask 的覆盖率（mask.mean()）。
        """
        image = image.convert("RGB")
        img_array = np.array(image)
        H, W = img_array.shape[:2]

        # ── Step 1: Qwen3-VL 提取 targets ─────────────────────────────────
        targets = self.extract_targets(image, question)

        # ── Step 2: SAM3 text prompt 分割，多 target logical_or 合并 ────────
        global_mask = np.zeros((H, W), dtype=bool)
        n_valid = 0

        for target in targets:
            # 每个 target 独立 set_image，避免不同 prompt 共享状态
            state  = self.sam3_processor.set_image(image)
            output = self.sam3_processor.set_text_prompt(
                state=state, prompt=target
            )
            masks = output["masks"]

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()

            # squeeze channel dim if shape (N, 1, H, W)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            # 空结果跳过（不做面积过滤）
            if len(masks) == 0:
                continue

            target_hit = False
            for mask in masks:
                mask_2d = mask > 0.5
                # resize 回原图尺寸（若 SAM3 输出与原图不一致）
                if mask_2d.shape != (H, W):
                    m = Image.fromarray((mask_2d * 255).astype(np.uint8))
                    m = m.resize((W, H), Image.NEAREST)
                    mask_2d = np.array(m) > 127
                global_mask = np.logical_or(global_mask, mask_2d)
                target_hit  = True

            if target_hit:
                n_valid += 1

        # ── Step 3: 像素 mask → CLIP 空间 → 24×24 patch mask ────────────
        mask_336   = _transform_mask_like_clip(global_mask, W, H, self.clip_size)
        patch_mask = _mask_to_patch_mask(mask_336, self.grid_h, self.grid_w)

        coverage = float(patch_mask.mean())

        return patch_mask, targets, n_valid, coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRASP extractor configuration helper.")
    parser.add_argument("--qwen-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qwen-device", default="cuda:1")
    parser.add_argument("--sam3-device", default="cuda:0")
    parser.add_argument("--sam3-repo", default=None, help="Optional SAM3 checkout path; also accepted via SAM3_REPO.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--clip-size", type=int, default=336)
    parser.add_argument("--grid-h", type=int, default=24)
    parser.add_argument("--grid-w", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sam3_repo:
        os.environ["SAM3_REPO"] = args.sam3_repo
    print("Instantiate GRASPExtractor with this configuration inside your dataset-building script.")
