# Data format

MVP keeps datasets and generated arrays outside Git. Paths in a sample manifest may be absolute or relative to the manifest's `train_dir`.

## Candidate-supervision layout

```text
train_data/
├── sample_list.json
├── train_inference/
│   ├── train_grasp.jsonl
│   ├── train_scope_64.jsonl
│   ├── train_scope_128.jsonl
│   ├── train_vispruner_64.jsonl
│   └── train_vispruner_128.jsonl
└── masks/
    └── <mask_key>/
        └── <sample_id>.npy
```

Each `sample_list.json` item must contain:

```json
{
  "uid": "unique-record-id",
  "sample_id": "stable-sample-id",
  "orig_id": "source-dataset-id",
  "image_path": "/path/to/image.jpg",
  "question": "What color is the sofa?",
  "answer": "blue",
  "subset": "gqa"
}
```

Each inference JSONL record is matched by `sample_id` and should include `subset`, `correct`, `n_kept`, `keep_ratio`, and `output`. Candidate mask arrays must have shape `(576,)`.

## Training manifest

`mvp.train` accepts either a list of samples or an object with a `samples` list and optional `train_dir`. Each normalized training item resolves these arrays:

| Field | Default relative path | Shape / dtype |
| --- | --- | --- |
| `hidden_path` | `hidden/<sample_id>.npy` | `(576, 1024)` float array; `(577, 1024)` is accepted and its CLS token is removed. |
| `question_tokens_path` | `question_tokens/<sample_id>.npz` | `input_ids` and `attention_mask`, both one-dimensional. |
| `oracle_mask_path` | `masks/oracle_min/<sample_id>.npy` | `(576,)` float soft-label array. |

The soft-label builder writes an explicit training manifest under `train_supervision/`; use that generated JSON for `--samples-json` whenever possible.

## Generated outputs

Training writes `config.json`, `history.json`, and checkpoint files under the selected output directory. lmms-eval writes one subdirectory per token budget. These outputs are ignored by Git and should be archived with the exact external dependency commit hashes used for the run.
