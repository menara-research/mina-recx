# Minang Embedder

Fine-tuned sentence embeddings for Minangkabau semantic search and cross-lingual retrieval.

This repository contains the local MinSTS-Retrieval benchmark data, training code, benchmark outputs, and figures.

The final exported model artifact is available locally but intentionally excluded from GitHub:

`/root/ling-proj/models/minang-embedder/jinaai_jina-embeddings-v5-text-nano-retrieval`

## Artifacts

| Path | Contents |
| --- | --- |
| `data/processed/benchmark.json` | MinSTS-Retrieval benchmark with monolingual retrieval, English-to-Minangkabau retrieval, STS, cross-lingual, and code-switching tasks. |
| `data/processed/training_pairs.json` | Generated positive, weak-positive, and negative training pairs. |
| `data/processed/training_with_negatives.json` | Contrastive training examples with BM25 hard negatives. |
| `data/processed/train_dataset_hf/` | Hugging Face dataset export used by training. |
| `results/` | Baseline, finetuned, and ablation benchmark JSON outputs. |
| `figures/` | Publication-style benchmark, ablation, heatmap, and training-loss plots as PNG and PDF. |
| `/root/ling-proj/models/minang-embedder/` | Local-only final exported SentenceTransformers model. The `models/` tree is ignored for GitHub. |

## Final Model Metrics

Source: `results/finetuned_minang-embedder.json`

| Metric | Value |
| --- | ---: |
| STS Spearman | 0.7975 |
| Min-En Accuracy@1 | 0.7825 |
| Min-ID Accuracy@1 | 0.9075 |
| Monolingual Recall@10 | 0.0410 |
| Monolingual MRR@10 | 0.0760 |
| Cross-En Recall@10 | 0.0455 |
| Code-switch cosine | 0.6736 |

## Ablation Summary

Source: `results/all_ablation_results.json`

| Model | STS Spearman | Min-En Acc@1 | Min-ID Acc@1 | Mono R@10 | Mono MRR@10 | Cross-En R@10 | Code-switch Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.4943 | 0.7025 | 0.9300 | 0.0400 | 0.0809 | 0.0510 | 0.7255 |
| epochs_3 | 0.7788 | 0.8650 | 0.9800 | 0.0340 | 0.0724 | 0.0385 | 0.7666 |
| epochs_5 | 0.7896 | 0.8400 | 0.9625 | 0.0360 | 0.0743 | 0.0400 | 0.7367 |
| epochs_7 | 0.7941 | 0.8075 | 0.9400 | 0.0380 | 0.0789 | 0.0485 | 0.7018 |
| epochs_10 | 0.7952 | 0.7775 | 0.9125 | 0.0400 | 0.0807 | 0.0420 | 0.6797 |
| temp_0.02 | 0.7760 | 0.7400 | 0.9275 | 0.0370 | 0.0682 | 0.0413 | 0.8201 |
| temp_0.1 | 0.7990 | 0.8750 | 0.9650 | 0.0395 | 0.0836 | 0.0393 | 0.7835 |
| temp_0.2 | 0.7992 | 0.8700 | 0.9450 | 0.0500 | 0.0902 | 0.0450 | 0.8618 |

## Figures

| Figure | File |
| --- | --- |
| Model comparison | `figures/performance_comparison.png`, `figures/performance_comparison.pdf` |
| Performance heatmap | `figures/performance_heatmap.png`, `figures/performance_heatmap.pdf` |
| Ablation study | `figures/ablation_study.png`, `figures/ablation_study.pdf` |
| Training loss | `figures/training_loss.png`, `figures/training_loss.pdf` |

## Run Locally

CPU steps (data prep and plotting) need only the default deps:

```bash
uv sync
uv run python src/data/prepare_data.py
```

GPU steps (training and evaluation) need the `gpu` extra for flash-attn:

```bash
uv sync --extra gpu
uv run python src/training/train.py
```

`scripts/eval_ablations.py` evaluates the ablation models under `models/ablations/`. Those models do not exist until you train them, one run per config, with `src/training/train_tracked.py`:

```bash
uv run python src/training/train_tracked.py --run-name epochs_3 --epochs 3
uv run python src/training/train_tracked.py --run-name epochs_5 --epochs 5
uv run python src/training/train_tracked.py --run-name epochs_7 --epochs 7
uv run python src/training/train_tracked.py --run-name epochs_10 --epochs 10
uv run python src/training/train_tracked.py --run-name temp_0.02 --temperature 0.02
uv run python src/training/train_tracked.py --run-name temp_0.1 --temperature 0.1
uv run python src/training/train_tracked.py --run-name temp_0.2 --temperature 0.2
uv run python scripts/eval_ablations.py
```

Plotting runs on CPU once results exist:

```bash
uv run python scripts/plot_results.py
```

Model weights, intermediate checkpoints, optimizer state, and local caches are excluded by `.gitignore`. Keep `/root/ling-proj/models/` as the local model artifact directory.
