#!/usr/bin/env python3
"""
Full pipeline: prepare data → baseline evaluation → train → evaluate trained models → compare.

Usage:
    uv run python src/run_pipeline.py --all
    uv run python src/run_pipeline.py --prepare-data
    uv run python src/run_pipeline.py --evaluate-baselines
    uv run python src/run_pipeline.py --train
    uv run python src/run_pipeline.py --evaluate-trained
    uv run python src/run_pipeline.py --compare
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

# Models to evaluate as baselines (pre-trained, no fine-tuning)
BASELINE_MODELS = [
    "jinaai/jina-embeddings-v5-text-nano",
    "intfloat/multilingual-e5-small",
    "LazarusNLP/all-indo-e5-small-v4",
    "LazarusNLP/all-NusaBERT-base-v4",
    "answerdotai/ModernBERT-base",
]

# Models to fine-tune
FINETUNE_MODELS = [
    "jinaai/jina-embeddings-v5-text-nano",
    "intfloat/multilingual-e5-small",
    "LazarusNLP/all-indo-e5-small-v4",
]


def disable_cudnn_sdp():
    """B300 has cuDNN version mismatch — disable, use flash SDPA instead."""
    torch.backends.cuda.enable_cudnn_sdp(False)


def step_prepare_data():
    """Step 1: Download and prepare training data + benchmark."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)
    
    from src.data.prepare_data import main as prepare_main
    prepare_main()


def step_evaluate_baselines(benchmark_path: str = "data/processed/benchmark.json"):
    """Step 2: Evaluate all baseline models on MinSTS-Retrieval."""
    logger.info("=" * 60)
    logger.info("STEP 2: Baseline Evaluation")
    logger.info("=" * 60)
    
    from src.benchmark.evaluate import MinSTSBenchmark, _convert_for_json
    from sentence_transformers import SentenceTransformer
    
    bench = MinSTSBenchmark(benchmark_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_id in BASELINE_MODELS:
        safe_name = model_id.replace("/", "_")
        result_path = RESULTS_DIR / f"baseline_{safe_name}.json"
        
        if result_path.exists():
            logger.info(f"Loading cached results for {model_id}")
            with open(result_path) as f:
                all_results[model_id] = json.load(f)
            continue
        
        logger.info(f"\nEvaluating baseline: {model_id}")

        is_jina = "jina" in model_id.lower()
        is_modernbert = "modernbert" in model_id.lower()
        
        try:
            if is_jina:
                model = SentenceTransformer(
                    model_id,
                    trust_remote_code=True,
                    model_kwargs={"dtype": torch.bfloat16},
                )
            elif is_modernbert:
                from sentence_transformers import models
                transformer = models.Transformer(model_id, trust_remote_code=True)
                pooling = models.Pooling(
                    transformer.get_word_embedding_dimension(),
                    pooling_mode="mean",
                )
                model = SentenceTransformer(modules=[transformer, pooling])
            else:
                model = SentenceTransformer(model_id)
            
            model.max_seq_length = 512
            
            results = bench.evaluate_all(model, model_id=model_id, batch_size=128)
            all_results[model_id] = results
            
            MinSTSBenchmark.print_results(results)

            with open(result_path, "w") as f:
                json.dump(_convert_for_json(results), f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to evaluate {model_id}: {e}")
            import traceback
            traceback.print_exc()

        del model
        torch.cuda.empty_cache()

    with open(RESULTS_DIR / "all_baselines.json", "w") as f:
        json.dump(_convert_for_json(all_results), f, indent=2, ensure_ascii=False)
    
    logger.info("Baseline evaluation complete!")
    return all_results


def step_train(data_path: str = "data/processed/training_with_negatives.json"):
    """Step 3: Fine-tune models."""
    logger.info("=" * 60)
    logger.info("STEP 3: Fine-tuning")
    logger.info("=" * 60)
    
    from src.training.train import train
    
    for model_id in FINETUNE_MODELS:
        safe_name = model_id.replace("/", "_")
        output_path = MODELS_DIR / "minang-embedder" / safe_name
        
        if output_path.exists() and (output_path / "config.json").exists():
            logger.info(f"Skipping {model_id} — already trained at {output_path}")
            continue
        
        logger.info(f"\nFine-tuning: {model_id}")
        train(
            model_id=model_id,
            data_path=data_path,
            output_dir="models/minang-embedder",
            epochs=10,
            batch_size=128,
            learning_rate=2e-5,
            temperature=0.05,
        )
    
    logger.info("Training complete!")


def step_evaluate_trained(benchmark_path: str = "data/processed/benchmark.json"):
    """Step 4: Evaluate fine-tuned models on MinSTS-Retrieval."""
    logger.info("=" * 60)
    logger.info("STEP 4: Trained Model Evaluation")
    logger.info("=" * 60)
    
    from src.benchmark.evaluate import MinSTSBenchmark, _convert_for_json
    from sentence_transformers import SentenceTransformer
    
    bench = MinSTSBenchmark(benchmark_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_id in FINETUNE_MODELS:
        safe_name = model_id.replace("/", "_")
        model_path = MODELS_DIR / "minang-embedder" / safe_name
        result_path = RESULTS_DIR / f"finetuned_{safe_name}.json"
        
        if not model_path.exists():
            logger.warning(f"Trained model not found: {model_path}")
            continue
        
        if result_path.exists():
            logger.info(f"Loading cached results for finetuned {model_id}")
            with open(result_path) as f:
                all_results[model_id] = json.load(f)
            continue
        
        logger.info(f"\nEvaluating fine-tuned: {model_id}")
        
        try:
            model = SentenceTransformer(str(model_path))
            results = bench.evaluate_all(model, model_id=f"finetuned/{model_id}", batch_size=128)
            all_results[model_id] = results
            
            MinSTSBenchmark.print_results(results)
            
            with open(result_path, "w") as f:
                json.dump(_convert_for_json(results), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()
        
        del model
        torch.cuda.empty_cache()
    
    with open(RESULTS_DIR / "all_finetuned.json", "w") as f:
        json.dump(_convert_for_json(all_results), f, indent=2, ensure_ascii=False)
    
    logger.info("Trained model evaluation complete!")
    return all_results


def step_compare():
    """Step 5: Compare baseline vs. fine-tuned results."""
    logger.info("=" * 60)
    logger.info("STEP 5: Comparison")
    logger.info("=" * 60)
    
    baseline_path = RESULTS_DIR / "all_baselines.json"
    finetuned_path = RESULTS_DIR / "all_finetuned.json"
    
    baselines = json.load(open(baseline_path)) if baseline_path.exists() else {}
    finetuned = json.load(open(finetuned_path)) if finetuned_path.exists() else {}
    
    if not baselines and not finetuned:
        logger.error("No results to compare!")
        return

    print("\n" + "=" * 80)
    print("MinSTS-Retrieval: Baseline vs. Fine-tuned Comparison")
    print("=" * 80)

    metrics = ["Recall@1", "Recall@10", "MRR@10", "nDCG@10"]

    for split_key, split_label in [("retrieval_monolingual", "Mono Retrieval"),
                                    ("retrieval_cross_en", "Cross-lingual Retrieval")]:
        print(f"\n{split_label}:")
        print(f"{'Model':<50} {'R@1':>8} {'R@10':>8} {'MRR@10':>8} {'nDCG@10':>8}")
        print("-" * 82)
        
        for model_id in set(list(baselines.keys()) + list(finetuned.keys())):
            if model_id in baselines and split_key in baselines[model_id]:
                b = baselines[model_id][split_key]
                row = f"{'baseline/' + model_id:<50}"
                for m in metrics:
                    row += f" {b.get(m, 0):.4f}  "
                print(row)

            if model_id in finetuned and split_key in finetuned[model_id]:
                f = finetuned[model_id][split_key]
                b_val = baselines.get(model_id, {}).get(split_key, {})
                row = f"{'finetuned/' + model_id:<50}"
                for m in metrics:
                    val = f.get(m, 0)
                    base = b_val.get(m, 0)
                    delta = val - base
                    sign = "+" if delta > 0 else ""
                    row += f" {val:.4f}  "
                print(row)
                row = f"{'  Δ':<50}"
                for m in metrics:
                    val = f.get(m, 0)
                    base = b_val.get(m, 0)
                    delta = val - base
                    sign = "+" if delta > 0 else ""
                    row += f" {sign}{delta:.4f}  "
                print(row)
    
    print(f"\nSTS (Spearman):")
    print(f"{'Model':<50} {'Spearman':>10} {'CI 95%':>20}")
    print("-" * 82)
    for model_id in set(list(baselines.keys()) + list(finetuned.keys())):
        for prefix, data in [("baseline", baselines), ("finetuned", finetuned)]:
            if model_id in data and "sts" in data[model_id]:
                s = data[model_id]["sts"]
                ci = s.get("Spearman_CI95", ["?", "?"])
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci[0], (int, float)) else str(ci)
                print(f"{prefix + '/' + model_id:<50} {s['Spearman']:>10.4f} {ci_str:>20}")
    
    print("=" * 80 + "\n")

    comparison = {"baselines": baselines, "finetuned": finetuned}
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(_convert_for_json(comparison), f, indent=2, ensure_ascii=False)


def _convert_for_json(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(x) for x in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--evaluate-baselines", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-trained", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    disable_cudnn_sdp()

    if args.all or args.prepare_data:
        step_prepare_data()
    
    if args.all or args.evaluate_baselines:
        step_evaluate_baselines()
    
    if args.all or args.train:
        step_train()
    
    if args.all or args.evaluate_trained:
        step_evaluate_trained()
    
    if args.all or args.compare:
        step_compare()
    
    if not any([args.all, args.prepare_data, args.evaluate_baselines, 
                args.train, args.evaluate_trained, args.compare]):
        parser.print_help()


if __name__ == "__main__":
    main()
