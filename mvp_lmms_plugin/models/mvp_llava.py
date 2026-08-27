from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
_lmms_root = Path(os.environ["LMMS_EVAL_REPO"]).expanduser().resolve() if os.environ.get("LMMS_EVAL_REPO") else ROOT / "lmmseval"
_llava_root = Path(os.environ["LLAVA_REPO"]).expanduser().resolve() if os.environ.get("LLAVA_REPO") else ROOT / "LLaVA"
for _path in [ROOT, _lmms_root, _llava_root]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lmms_eval.api.instance import Instance
from lmms_eval.models.simple.llava import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    Llava,
    conv_templates,
    process_images,
    tokenizer_image_token,
)
from loguru import logger as eval_logger

from mvp.model import (
    MVPPruner,
    model_config_from_checkpoint as mvp_model_config_from_checkpoint,
)


class _ProjectorPruningHook:
    def __init__(self, llava_model) -> None:
        self._keep: torch.Tensor | None = None
        self._handle = llava_model.model.mm_projector.register_forward_pre_hook(self._hook_fn)

    def set_keep_indices(self, indices: torch.Tensor) -> int:
        indices = indices.detach().to(dtype=torch.long, device="cpu")
        self._keep = indices
        return int(indices.numel())

    def clear(self) -> None:
        self._keep = None

    def remove(self) -> None:
        self._handle.remove()

    def _hook_fn(self, module, inputs):
        if self._keep is None:
            return inputs

        x = inputs[0]
        idx = self._keep.to(x.device)
        if x.dim() == 2:
            idx = idx[idx < x.shape[0]]
            pruned = x[idx, :]
        else:
            idx = idx[idx < x.shape[1]]
            pruned = x[:, idx, :]
        proj_device = next(module.parameters()).device
        pruned = pruned.to(proj_device)
        return (pruned,) + inputs[1:]


def _build_mvp_model_from_checkpoint(ckpt: dict) -> torch.nn.Module:
    model_config = ckpt.get("model_config") or {}
    args = ckpt.get("args") or {}
    nested_arg_config = args.get("model_config") or {}
    block_type = str(
        model_config.get("block_type")
        or nested_arg_config.get("block_type")
        or args.get("model_variant")
        or "base"
    )

    if block_type == "self_cross_swiglu":
        return MVPPruner.from_config(mvp_model_config_from_checkpoint(ckpt))
    raise ValueError(
        f"Unsupported MVP checkpoint block_type/model_variant for lmms-eval: {block_type}"
    )


class MVPLlava(Llava):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        adapter_ckpt: str = "outputs/mvp_pruner/checkpoints/best.pt",
        selection_mode: str = "topk",
        topk: int = 64,
        threshold: float = 0.5,
        min_keep_tokens: int = 1,
        max_keep_tokens: int = 576,
        mvp_dtype: str = "float32",
        conv_template: str = "vicuna_v1",
        batch_size: int | str = 1,
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        truncate_context: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            conv_template=conv_template,
            device=device,
            device_map=device_map,
            truncate_context=truncate_context,
            use_cache=use_cache,
            **kwargs,
        )

        if selection_mode not in {"topk", "threshold"}:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

        self.pretrained = pretrained
        self.adapter_ckpt = str(Path(adapter_ckpt).resolve())
        self.selection_mode = selection_mode
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.min_keep_tokens = max(1, int(min_keep_tokens))
        self.max_keep_tokens = max(1, int(max_keep_tokens))
        self.mvp_dtype = self._parse_dtype(mvp_dtype)

        ckpt = torch.load(self.adapter_ckpt, map_location="cpu")
        self.mvp_pruner = _build_mvp_model_from_checkpoint(ckpt)
        self.mvp_pruner.load_state_dict(ckpt["model"], strict=True)
        self.mvp_pruner.eval()
        self.mvp_pruner.to(device=self.device, dtype=self.mvp_dtype)
        self.mvp_pruner.requires_grad_(False)
        self.mvp_block_type = str(
            (ckpt.get("model_config") or {}).get("block_type")
            or ((ckpt.get("args") or {}).get("model_config") or {}).get("block_type")
            or (ckpt.get("args") or {}).get("model_variant")
            or "self_cross_swiglu"
        )

        self.pruning_hook = _ProjectorPruningHook(self.model)

        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(device=self.device, dtype=torch.float16)

        self._warned_multi_image = False
        self._warned_no_visual = False

    @staticmethod
    def _parse_dtype(dtype: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported mvp_dtype: {dtype}")
        return mapping[dtype]

    def _extract_question_text(self, context: str) -> str:
        text = str(context).replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if not text:
            raise ValueError(f"Empty question text after cleaning: {context!r}")
        return text

    def _normalize_visuals(self, visuals) -> List[Image.Image]:
        if visuals is None:
            return []
        if not isinstance(visuals, list):
            visuals = [visuals]

        out: List[Image.Image] = []
        for visual in visuals:
            if isinstance(visual, str):
                out.append(Image.open(visual).convert("RGB"))
            elif hasattr(visual, "convert"):
                out.append(visual.convert("RGB"))
            else:
                raise TypeError(f"Unsupported visual type: {type(visual)}")
        return out

    def _build_prompt(self, context: str, num_images: int) -> str:
        if num_images > 0 and DEFAULT_IMAGE_TOKEN not in context:
            image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
            question = image_tokens + "\n" + context
        else:
            question = context

        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.no_grad()
    def _compute_visual_tokens(self, image: Image.Image) -> torch.Tensor:
        image_tensor = process_images([image], self._image_processor, self._config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        vision_tower = self.model.get_model().get_vision_tower()
        clip_model = vision_tower.vision_tower
        clip_device = next(clip_model.parameters()).device
        clip_dtype = next(clip_model.parameters()).dtype
        image_tensor = image_tensor.to(device=clip_device, dtype=clip_dtype)

        outputs = clip_model(
            image_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
        visual_tokens = outputs.hidden_states[vision_tower.select_layer][:, 1:, :]
        return visual_tokens.to(device=self.device, dtype=self.mvp_dtype)

    @torch.no_grad()
    def _compute_question_embeds(self, question_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            question_text,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        question_embeds = self.model.get_model().embed_tokens(input_ids).to(dtype=self.mvp_dtype)
        return question_embeds, attention_mask

    @torch.no_grad()
    def _select_keep_indices(self, image: Image.Image, question_text: str) -> torch.Tensor:
        visual_tokens = self._compute_visual_tokens(image)
        question_embeds, attention_mask = self._compute_question_embeds(question_text)
        probs = self.mvp_pruner.predict_proba(
            visual_tokens=visual_tokens,
            question_tokens=question_embeds,
            question_attention_mask=attention_mask,
        )[0]

        if self.selection_mode == "topk":
            k = min(max(self.min_keep_tokens, self.topk), min(self.max_keep_tokens, probs.numel()))
            keep = torch.topk(probs, k=k, largest=True).indices
            keep = torch.sort(keep).values
        else:
            keep = torch.nonzero(probs >= self.threshold, as_tuple=False).squeeze(-1)
            if keep.numel() < self.min_keep_tokens:
                keep = torch.topk(probs, k=self.min_keep_tokens, largest=True).indices
            if keep.numel() > self.max_keep_tokens:
                keep = torch.topk(probs, k=self.max_keep_tokens, largest=True).indices
            keep = torch.sort(keep).values

        return keep

    def _prepare_generation_kwargs(self, gen_kwargs: dict, visuals: List[Image.Image]) -> dict:
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 128)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)
        gen_kwargs["image_sizes"] = [visual.size for visual in visuals]
        return gen_kwargs

    def _generate_single(self, context: str, visuals: List[Image.Image], gen_kwargs: dict) -> str:
        prompt = self._build_prompt(context=context, num_images=len(visuals))
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_id).to(self.device)

        if visuals:
            image_tensor = process_images(visuals, self._image_processor, self._config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(dtype=torch.float16, device=self.device) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
        else:
            image_tensor = None

        until = [self.tok_decode(self.eot_token_id)]
        if "until" in gen_kwargs:
            until = gen_kwargs.pop("until")
            if isinstance(until, str):
                until = [until]

        with torch.inference_mode():
            do_sample = gen_kwargs["temperature"] > 0
            generate_kwargs = {
                "attention_mask": attention_mask,
                "pad_token_id": pad_token_id,
                "images": image_tensor,
                "image_sizes": gen_kwargs.get("image_sizes"),
                "do_sample": do_sample,
                "num_beams": gen_kwargs["num_beams"],
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
            }
            if do_sample:
                generate_kwargs["temperature"] = gen_kwargs["temperature"]
                if gen_kwargs["top_p"] is not None:
                    generate_kwargs["top_p"] = gen_kwargs["top_p"]

            output_ids = self.model.generate(
                input_ids,
                **generate_kwargs,
            )

        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        for term in until:
            if term:
                text = text.split(term)[0]
        return text.strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        desc = f"Model Responding [{self.selection_mode}]"
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc=desc)

        for req in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            context = contexts[0] if isinstance(contexts, list) else contexts
            doc = self.task_dict[task][split][doc_id]
            visuals = self._normalize_visuals(doc_to_visual(doc))
            gen_kwargs = self._prepare_generation_kwargs(gen_kwargs, visuals)

            use_mvp = len(visuals) == 1
            if len(visuals) > 1 and not self._warned_multi_image and self.rank == 0:
                eval_logger.warning("mvp_llava currently prunes only single-image samples; multi-image samples fall back to vanilla LLaVA.")
                self._warned_multi_image = True
            if len(visuals) == 0 and not self._warned_no_visual and self.rank == 0:
                eval_logger.warning("mvp_llava received samples without visuals; those samples fall back to vanilla LLaVA.")
                self._warned_no_visual = True

            if use_mvp:
                question_text = self._extract_question_text(context)
                keep_indices = self._select_keep_indices(visuals[0], question_text)
                self.pruning_hook.set_keep_indices(keep_indices)

            try:
                output = self._generate_single(
                    context=context,
                    visuals=visuals,
                    gen_kwargs=gen_kwargs,
                )
            finally:
                if use_mvp:
                    self.pruning_hook.clear()

            res.append(output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), output)
            pbar.update(1)

        pbar.close()
        return res
