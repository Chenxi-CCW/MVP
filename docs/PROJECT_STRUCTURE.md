# Project structure

This document explains every source-controlled file in the MVP repository. Runtime outputs and third-party repositories are intentionally excluded.

## Root files

| Path | Purpose |
| --- | --- |
| `.gitattributes` | Keeps text files on consistent LF line endings and marks figures as binary. |
| `.gitignore` | Prevents local environments, third-party checkouts, datasets, checkpoints, and generated outputs from being committed. |
| `LICENSE` | Apache License 2.0 terms. |
| `README.md` | Project overview, installation, workflow, limitations, and citation guidance. |
| `requirements.txt` | Runtime Python dependencies installed from PyPI. |
| `requirements-dev.txt` | Runtime dependencies plus the test runner. |

## MVP training and inference

| Path | Purpose |
| --- | --- |
| `mvp/__init__.py` | Exposes the main dataset, embedder, and pruner APIs. |
| `mvp/model.py` | Defines `MVPPruner`, its attention blocks, and checkpoint configuration helpers. |
| `mvp/data.py` | Loads sample manifests, resolves feature paths, splits data, and batches training tensors. |
| `mvp/question_embedder.py` | Loads a frozen LLaVA language model and converts question token IDs into embeddings. |
| `mvp/train.py` | Command-line training loop, BCE/ranking losses, metrics, scheduling, and checkpoint writing. |
| `mvp/run_lmms_eval.py` | Launches lmms-eval with the MVP plugin for several token-retention budgets. |

## Supervision generation

| Path | Purpose |
| --- | --- |
| `mask_extraction/__init__.py` | Marks the supervision code as a Python package. |
| `mask_extraction/extractors/__init__.py` | Marks the extractor directory as a Python package. |
| `mask_extraction/extractors/grasp.py` | Uses Qwen3-VL and SAM3 to turn question-relevant objects into a 576-patch mask. |
| `mask_extraction/extractors/scope_extractor.py` | Reproduces SCOPE diversity-based greedy patch selection. |
| `mask_extraction/extractors/vispruner_extractor.py` | Reproduces VisPruner attention/diversity patch selection. |
| `mask_extraction/dataset_tools/__init__.py` | Marks the dataset-tool directory as a Python package. |
| `mask_extraction/dataset_tools/utils.py` | Cleans/tokenizes questions, persists token arrays, validates manifests, and provides dataset helpers. |
| `mask_extraction/dataset_tools/build_soft_labels.py` | Computes success/failure reliability and fuses candidate masks into SSF family-beta soft labels. |

## lmms-eval integration

| Path | Purpose |
| --- | --- |
| `mvp_lmms_plugin/__init__.py` | Declares the custom lmms-eval plugin package. |
| `mvp_lmms_plugin/models/__init__.py` | Registers the `mvp_llava` model name. |
| `mvp_lmms_plugin/models/mvp_llava.py` | Loads an MVP checkpoint, scores image patches, prunes before the multimodal projector, and implements lmms-eval generation. |

## Documentation, tests, and figures

| Path | Purpose |
| --- | --- |
| `docs/PROJECT_STRUCTURE.md` | This file-by-file map. |
| `docs/DATA_FORMAT.md` | Input directory and manifest contracts for supervision and training. |
| `tests/test_soft_label_utils.py` | Lightweight unit tests for mask-family inference and weighted soft-mask fusion. |
| `assets/data_pipeline.png` | Figure showing the three supervision views and their selected visual tokens. |
| `assets/model_architecture.png` | Figure showing MVPPruner between the frozen encoders and frozen LLM decoder. |
