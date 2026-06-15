#!/usr/bin/env python3
"""Evaluate all ablation models + baseline on MinSTS benchmark."""

import json
import os
import torch
from sentence_transformers import SentenceTransformer
from src.benchmark.evaluate import MinSTSBenchmark, _convert_for_json

torch.backends.cuda.enable_cudnn_sdp(False)

bench = MinSTSBenchmark('data/processed/benchmark.json')

# Models to evaluate
models_to_eval = [
    ("baseline", "jinaai/jina-embeddings-v5-text-nano-retrieval"),
    ("epochs_3", "models/ablations/epochs_3"),
    ("epochs_5", "models/ablations/epochs_5"),
    ("epochs_7", "models/ablations/epochs_7"),
    ("epochs_10", "models/ablations/epochs_10"),
    ("temp_0.02", "models/ablations/temp_0.02"),
    ("temp_0.1", "models/ablations/temp_0.1"),
    ("temp_0.2", "models/ablations/temp_0.2"),
]

all_results = {}

for name, path in models_to_eval:
    print(f"\nEvaluating: {name} ({path})")
    if path.startswith("models/") and not os.path.isdir(path):
        print(f"  SKIP: {path} not found. Train this ablation first "
              f"(see README) or remove it from models_to_eval.")
        continue
    try:
        model = SentenceTransformer(path, trust_remote_code=True)
        model.max_seq_length = 512
        results = bench.evaluate_all(model, model_id=name, batch_size=128)
        all_results[name] = results
        
        # Save individual
        out_path = f"results/ablation_{name}.json"
        with open(out_path, 'w') as f:
            json.dump(_convert_for_json(results), f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")
        
        # Print key metrics
        r = results
        mono_r10 = r.get('retrieval_monolingual', {}).get('Recall@10', 0)
        sts = r.get('sts', {}).get('Spearman', 0)
        en_acc = r.get('cross_lingual', {}).get('min_en', {}).get('Accuracy@1', 0)
        id_acc = r.get('cross_lingual', {}).get('min_id', {}).get('Accuracy@1', 0)
        cs = r.get('codeswitch', {}).get('avg_cosine_similarity', 0)
        print(f"  R@10={mono_r10:.4f}, STS={sts:.4f}, en@1={en_acc:.4f}, id@1={id_acc:.4f}, CS={cs:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

# Save combined results
with open("results/all_ablation_results.json", 'w') as f:
    json.dump({k: _convert_for_json(v) for k, v in all_results.items()}, f, indent=2, ensure_ascii=False)
print("\nSaved combined results to results/all_ablation_results.json")
