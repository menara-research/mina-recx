# Minang Embedder

Minang Embedder is a Minangkabau sentence-embedding model and benchmark repo for semantic search, cross-lingual retrieval, and code-switch evaluation.

It fine-tunes `jinaai/jina-embeddings-v5-text-nano-retrieval` on NusaX-derived Minangkabau, Indonesian, and English pairs, then evaluates the result on a local constructed benchmark: MinSTS-Retrieval.

It is not a hosted service, generic model zoo, or manually human-annotated STS benchmark.

Model weights:

https://huggingface.co/apsys/minang-embedder

Code, benchmark artifacts, result JSONs, and figures live in this repo.

## Quickstart

Install and embed text with the public Hugging Face model:

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("apsys/minang-embedder", trust_remote_code=True)

queries = ["Paliang suko bana makan siang di siko ayam jo ladonyo lamak bana."]
docs = [
    "I love having lunch here because the chicken and sambal are delicious.",
    "The train ticket was booked yesterday.",
    "Kurang pas kalau bakunjuang ka banduang tanpa mancicipi batagor.",
]

q = model.encode_query(queries, normalize_embeddings=True)
d = model.encode_document(docs, normalize_embeddings=True)
print(model.similarity(q, d))
```

Expected result: the English lunch/sambal translation should rank above unrelated text.

## Smallest Real Example

Inspect the committed benchmark result:

```bash
jq '.sts.Spearman, .cross_lingual.min_en["Accuracy@1"], .codeswitch.avg_cosine_similarity' \
  results/finetuned_minang-embedder.json
```

Expected output:

```txt
0.7974851829791335
0.7825
0.6736049652099609
```

## Reproduce

The repo uses `uv`.

CPU steps need only default dependencies:

```bash
uv sync
uv run python src/data/prepare_data.py
uv run python scripts/plot_results.py
```

GPU steps need the `gpu` extra for `flash-attn`:

```bash
uv sync --extra gpu
uv run python src/training/train.py
```

Run ablations before evaluating `models/ablations/`:

```bash
uv run python src/training/train_tracked.py --run-name epochs_3 --epochs 3
uv run python src/training/train_tracked.py --run-name epochs_5 --epochs 5
uv run python src/training/train_tracked.py --run-name epochs_7 --epochs 7
uv run python src/training/train_tracked.py --run-name epochs_10 --epochs 10
uv run python src/training/train_tracked.py --run-name temp_0.02 --temperature 0.02
uv run python src/training/train_tracked.py --run-name temp_0.1 --temperature 0.1
uv run python src/training/train_tracked.py --run-name temp_0.2 --temperature 0.2
uv run python scripts/eval_ablations.py
uv run python scripts/plot_results.py
```

The local final model export is intentionally ignored by Git:

```txt
/root/ling-proj/models/minang-embedder/jinaai_jina-embeddings-v5-text-nano-retrieval/
```

Use the public HF model for normal loading:

```txt
apsys/minang-embedder
```

## Results

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

Best ablation summary from `results/all_ablation_results.json`:

| Model | STS Spearman | Min-En Acc@1 | Min-ID Acc@1 | Mono R@10 | Mono MRR@10 | Cross-En R@10 | Code-switch Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.4943 | 0.7025 | 0.9300 | 0.0400 | 0.0809 | 0.0510 | 0.7255 |
| temp_0.2 | 0.7992 | 0.8700 | 0.9450 | 0.0500 | 0.0902 | 0.0450 | 0.8618 |
| final export | 0.7975 | 0.7825 | 0.9075 | 0.0410 | 0.0760 | 0.0455 | 0.6736 |

Full baselines and ablations are in `results/`.

## Comparison

Minang Embedder is designed for a narrow low-resource retrieval wedge: Minangkabau text, Minangkabau-English alignment, Minangkabau-Indonesian alignment, and code-switch robustness.

| Model or artifact | Best for | Tradeoff |
| --- | --- | --- |
| `apsys/minang-embedder` | Minangkabau semantic search and local NusaX-derived evaluation | Small domain-specific fine-tune; not a broad multilingual leaderboard model |
| `jinaai/jina-embeddings-v5-text-nano-retrieval` | Compact multilingual retrieval baseline | Strong general base, but not specialized for the constructed Minangkabau benchmark |
| `intfloat/multilingual-e5-small` | General multilingual retrieval with a common prefix convention | Broader model; less targeted to this Minangkabau data path |
| `LazarusNLP/all-indo-e5-small-v4` | Indonesian-focused embeddings | Indonesian strength does not automatically cover Minangkabau transfer |
| `LazarusNLP/all-NusaBERT-base-v4` | Nusa-language representation baseline | Raw representation baseline; not optimized here as a retrieval SentenceTransformer |

Use this repo for:

- inspecting a concrete Minangkabau embedding fine-tune
- reproducing NusaX-derived training/eval artifacts
- comparing baselines on the same local benchmark JSON
- adapting the benchmark construction code for another Indonesian local language

Do not use it as:

- a claim of state-of-the-art Minangkabau STS
- a production relevance benchmark
- a hosted embedding API
- a general multilingual embedding replacement

## Benchmark

MinSTS-Retrieval is generated in `src/data/prepare_data.py` from:

- `mteb/NusaXBitextMining`: `eng-min`, `eng-ind`
- `mteb/nusa_x_senti`: `min`, `ind`, `eng`

Benchmark artifact:

```txt
data/processed/benchmark.json
```

| Section | Construction | Count |
| --- | --- | ---: |
| Monolingual retrieval | Minangkabau sentiment test text retrieves same-label Minangkabau texts | 400 queries, 400 docs |
| Cross EN to MIN retrieval | English sentiment test text retrieves same-label Minangkabau texts | 400 queries |
| STS | Translation, same-sentiment, and different-sentiment pairs with heuristic scores | 731 pairs |
| Cross-lingual bitext | Shared NusaX IDs for Minangkabau-English and Minangkabau-Indonesian | 400 each |
| Code-switching | Synthetic Minangkabau/Indonesian mixed text paired with original Minangkabau or English | 100 examples |

Metrics are computed in `src/benchmark/evaluate.py`:

| Task | Metrics |
| --- | --- |
| Retrieval | Recall@1, Recall@10, MRR@10, nDCG@10 |
| STS | Spearman correlation |
| Cross-lingual bitext | Accuracy@1, MRR, Recall@5, Recall@10 |
| Code-switching | average cosine similarity, Spearman |

Reproduce benchmark artifacts:

```bash
uv run python src/data/prepare_data.py
uv run python scripts/eval_ablations.py
```

## Mental Model

The repo has three layers:

```txt
data/processed/      committed benchmark and training artifacts
src/                 data preparation, training, and benchmark code
results/ + figures/  visible proof: JSON metrics and rendered plots
```

Pipeline:

```txt
NusaX bitext + NusaX sentiment
  -> generated training pairs
  -> BM25 hard negatives
  -> SentenceTransformers fine-tune
  -> MinSTS-Retrieval benchmark
  -> result JSONs
  -> plots
```

## Supported Paths

| Path | Status |
| --- | --- |
| Load public model from HF | supported |
| Recreate local benchmark JSON | supported |
| Re-run baseline and ablation evaluation | supported |
| Re-plot figures from committed results | supported |
| Train the local model path from scratch | supported with local GPU/runtime assumptions |
| Upload checkpoints to GitHub | not supported; model files are intentionally ignored |
| Treat MinSTS-Retrieval as human STS gold data | not supported |

## Repo Layout

```txt
.
  README.md
  ARTIFACTS.md
  pyproject.toml
  uv.lock

  configs/
    datasets.yaml

  data/processed/
    benchmark.json
    training_pairs.json
    training_with_negatives.json
    train_dataset_hf/

  src/
    data/prepare_data.py
    training/train.py
    training/train_tracked.py
    benchmark/evaluate.py
    run_pipeline.py

  scripts/
    eval_ablations.py
    plot_results.py

  results/
    *.json

  figures/
    *.png
    *.pdf
```

## Docs

Start here:

- Model card: https://huggingface.co/apsys/minang-embedder
- Artifact inventory: `ARTIFACTS.md`
- Benchmark construction: `src/data/prepare_data.py`
- Evaluation metrics: `src/benchmark/evaluate.py`
- Plot generation: `scripts/plot_results.py`

## License And Attribution

The public model is released as `cc-by-nc-sa-4.0` because it derives from:

- `jinaai/jina-embeddings-v5-text-nano-retrieval`, listed as `cc-by-nc-4.0`
- NusaX-derived datasets, listed as `cc-by-sa-4.0`

Data sources:

- https://huggingface.co/datasets/mteb/NusaXBitextMining
- https://huggingface.co/datasets/mteb/nusa_x_senti
- https://huggingface.co/datasets/indonlp/NusaX-senti

Primary NusaX reference:

```bibtex
@misc{winata2022nusax,
  title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
  author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony, Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo, Radityo Eko and Fung, Pascale and Baldwin, Timothy and Lau, Jey Han and Sennrich, Rico and Ruder, Sebastian},
  year={2022},
  eprint={2205.15960},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Limitations

- The benchmark is constructed from NusaX labels and alignments, not manually annotated Minangkabau STS.
- Retrieval relevance is approximated through sentiment labels, not human search judgments.
- STS scores are heuristic: translation is high similarity, same sentiment is medium/high, different sentiment is low.
- The model inherits constraints from the Jina base model and NusaX-derived data.
- The GitHub repo excludes model weights and checkpoints; use `apsys/minang-embedder` on Hugging Face for weights.
- Production retrieval quality should be validated on domain-specific Minangkabau queries and documents.
