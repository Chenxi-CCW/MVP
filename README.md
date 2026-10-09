<div align="center">

# MVP: Multi-View-Guided Token Pruning for Vision-Language Models

**EMNLP 2026 Main Conference**

Chenxi Wang · Dexing Zhong · Siyu Shi · Bowen Ping · Hang Yan · Huayang Li · Huikai Shao

[![Conference](https://img.shields.io/badge/EMNLP_2026-Main_Conference-235aa6?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-555?style=flat-square)](LICENSE)

Official implementation of **MVP**, a multi-view framework for efficient visual-token pruning in vision-language models.

</div>

MVP trains a lightweight, question-conditioned pruner to score the 576 visual patch tokens produced by LLaVA-1.5. Its supervision combines three complementary views of useful visual evidence:

- **Question-guided evidence** from GRASP object masks (Qwen3-VL + SAM3)
- **Diversity-based evidence** from SCOPE
- **Attention-based evidence** from VisPruner

Success/failure-aware fusion turns these candidate masks into soft supervision. At inference time, MVP keeps the highest-scoring tokens before LLaVA's multimodal projector.

![MVP supervision pipeline](assets/data_pipeline.png?raw=1)

![MVP model architecture](assets/model_architecture.png?raw=1)

## Repository status

This repository contains the training, supervision-building, and lmms-eval integration code. Datasets, third-party model repositories, pretrained model weights, and MVP checkpoints are not bundled.

The current implementation targets single-image LLaVA-1.5 inputs with a `24 x 24` patch grid (`576` patch tokens). Multi-image and text-only samples fall back to vanilla LLaVA during evaluation.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Chenxi-CCW/MVP.git
cd MVP

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The full pipeline also expects compatible local checkouts of:

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- SAM3 (for GRASP mask extraction)

Set their locations explicitly when they are not stored at the repository root:

```bash
export LLAVA_REPO=/path/to/LLaVA
export LMMS_EVAL_REPO=/path/to/lmms-eval
export SAM3_REPO=/path/to/sam3
```

`flash-attn` is optional. GRASP automatically falls back to the default Transformers attention implementation when it is unavailable.

## Workflow

### 1. Build candidate masks

GRASP converts a question and image into a question-guided patch mask:

```python
from PIL import Image
from mask_extraction.extractors.grasp import GRASPExtractor

extractor = GRASPExtractor(
    qwen_path="Qwen/Qwen3-VL-8B-Instruct",
    qwen_device="cuda:1",
    sam3_device="cuda:0",
)
mask, targets, num_valid_targets, coverage = extractor.compute_mask(
    Image.open("example.jpg"),
    "What color is the sofa?",
)
```

SCOPE and VisPruner operate on the penultimate CLIP hidden state and attention map:

```python
from mask_extraction.extractors.scope_extractor import SCOPEExtractor
from mask_extraction.extractors.vispruner_extractor import VisPrunerExtractor

scope_masks = SCOPEExtractor(token_counts=(64, 128)).compute_masks(hidden_pen, attn_pen)
vispruner_masks = VisPrunerExtractor(token_counts=(64, 128)).compute_masks(hidden_pen, attn_pen)
```

Here, `hidden_pen` has shape `(577, D)` (CLS + patches) and `attn_pen` has shape `(heads, 577, 577)`.

### 2. Fuse masks into soft labels

```bash
python -m mask_extraction.dataset_tools.build_soft_labels \
  --train-dir /path/to/train_data \
  --samples-json /path/to/train_data/sample_list.json
```

The builder reads candidate inference results from `train_inference/*.jsonl`, loads their masks, and writes fused masks plus manifests under `soft_masks/`, `records/`, and `train_supervision/`. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the expected layout and schema.

### 3. Train MVPPruner

```bash
python -m mvp.train \
  --samples-json /path/to/train_data/train_supervision/ssf_family_beta_soft_label_samples.json \
  --llava-path /path/to/llava-v1.5-7b \
  --output-dir outputs/mvp_pruner
```

The best and latest checkpoints are saved to `outputs/mvp_pruner/checkpoints/`.

### 4. Evaluate with lmms-eval

```bash
python -m mvp.run_lmms_eval \
  --tasks pope,gqa \
  --pretrained /path/to/llava-v1.5-7b \
  --adapter-ckpt outputs/mvp_pruner/checkpoints/best.pt \
  --lmms-eval-repo /path/to/lmms-eval \
  --llava-repo /path/to/LLaVA
```

The runner evaluates top-k settings of 32, 64, and 128 retained visual tokens. Additional lmms-eval arguments can be passed with `--extra-eval-args`.

## Code map

- [`mvp/`](mvp/) contains the pruner, dataset, training loop, question embedder, and evaluation launcher.
- [`mask_extraction/`](mask_extraction/) contains candidate-mask extractors and soft-label construction.
- [`mvp_lmms_plugin/`](mvp_lmms_plugin/) contains the custom lmms-eval LLaVA integration.
- [`assets/`](assets/) contains the paper overview figures.
- [`tests/`](tests/) contains lightweight tests for the soft-label utilities.

A file-by-file explanation is available in [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Reproducibility notes

- Training defaults to seed `42` and stores the complete configuration in `outputs/mvp_pruner/config.json`.
- Checkpoints include model configuration, optimizer state, scheduler state, epoch, and validation metrics.
- Large generated artifacts (`*.pt`, `*.npy`, datasets, checkpoints, and evaluation outputs) are excluded from Git by default.
- External repositories evolve independently; record their commit hashes alongside experiment results.

## Citation

If you use this work, please cite the MVP paper. This entry will be updated with the official ACL Anthology metadata after the proceedings are released.

```bibtex
@inproceedings{wang2026mvp,
  title     = {MVP: Multi-View-Guided Token Pruning for Vision-Language Models},
  author    = {Wang, Chenxi and Zhong, Dexing and Shi, Siyu and Ping, Bowen and Yan, Hang and Li, Huayang and Shao, Huikai},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

Released under the [Apache License 2.0](LICENSE).
