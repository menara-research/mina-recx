# Artifact Inventory

## Committed Artifact Groups

| Group | Path | Notes |
| --- | --- | --- |
| Benchmark data | `data/processed/benchmark.json` | Local MinSTS-Retrieval benchmark. |
| Training data | `data/processed/training_pairs.json` | Generated pair data. |
| Training data | `data/processed/training_with_negatives.json` | Contrastive examples with hard negatives. |
| Training data | `data/processed/train_dataset_hf/` | Hugging Face dataset export. |
| Benchmark outputs | `results/*.json` | Baselines, finetuned model, and ablation results. |
| Tables | `README.md` | Final metrics and ablation summary tables. |
| Graphs | `figures/*.png`, `figures/*.pdf` | Comparison, heatmap, ablation, and loss figures. |
| Source | `src/`, `scripts/`, `configs/` | Data prep, training, evaluation, plotting, and dataset configuration. |

## Local-Only Artifact

| Group | Local path | Notes |
| --- | --- | --- |
| Model | `/root/ling-proj/models/minang-embedder/jinaai_jina-embeddings-v5-text-nano-retrieval/` | Final exported SentenceTransformers model. Excluded from GitHub. |

## Excluded Local State

The following are intentionally ignored:

| Path or pattern | Reason |
| --- | --- |
| `.venv/` | Local virtual environment. |
| `.bg-shell/`, `.gsd/` | Agent/runtime state. |
| `data/cache/` | Regenerable dataset cache. |
| `models/` | Local model exports and checkpoints; result JSONs are retained in `results/`. |
| `*.pt`, `*.pth` | Optimizer/RNG/checkpoint state. |
