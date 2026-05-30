#!/usr/bin/env python3
"""
MinSTS-Retrieval: First retrieval/STS benchmark for Minangkabau.

Handles different model APIs:
- Jina v5: model.encode(texts=..., task=..., prompt_name=...) with task-specific prompts
- multilingual-e5: requires "query: "/"passage: " prefix
- LazarusNLP/NusaBERT: standard sentence-transformers encode()
- ModernBERT: raw MLM, needs pooling wrapper for embeddings
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─── Model-specific encode helpers ───────────────────────────────────

def encode_with_model(model, texts: list[str], batch_size: int = 128, 
                      task: str = "retrieval", role: str = "document",
                      model_id: str = "") -> np.ndarray:
    """
    Encode texts using the correct API for each model type.
    
    Handles:
    - Jina v5: task + prompt_name kwargs
    - multilingual-e5: "query: "/"passage: " prefixes
    - LazarusNLP: plain encode()
    - ModernBERT (raw): mean-pooling over last hidden state
    """
    # Detect model family
    is_jina = "jina" in model_id.lower()
    is_e5 = "e5" in model_id.lower() and "jina" not in model_id.lower()
    
    if is_jina:
        # Jina v5: task-aware encoding
        prompt_name = "query" if role == "query" else "document"
        return model.encode(
            sentences=texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            task=task,
            prompt_name=prompt_name,
        )
    elif is_e5:
        # multilingual-e5: prefix convention
        prefix = "query: " if role == "query" else "passage: "
        prefixed = [prefix + t for t in texts]
        return model.encode(
            sentences=prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    else:
        # Standard sentence-transformers encode (LazarusNLP, NusaBERT, etc.)
        return model.encode(
            sentences=texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )


def encode_sts_with_model(model, texts1: list[str], texts2: list[str],
                          batch_size: int = 128, model_id: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Encode STS pairs — symmetric task, use text-matching for Jina."""
    is_jina = "jina" in model_id.lower()
    is_e5 = "e5" in model_id.lower() and "jina" not in model_id.lower()
    
    if is_jina:
        emb1 = model.encode(texts1, batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=True, task="text-matching")
        emb2 = model.encode(texts2, batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=True, task="text-matching")
    elif is_e5:
        # Symmetric task: both use "query: " prefix
        prefixed1 = ["query: " + t for t in texts1]
        prefixed2 = ["query: " + t for t in texts2]
        emb1 = model.encode(prefixed1, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        emb2 = model.encode(prefixed2, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    else:
        emb1 = model.encode(texts1, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        emb2 = model.encode(texts2, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    
    return emb1, emb2


# ─── Benchmark class ──────────────────────────────────────────────────

class MinSTSBenchmark:
    """MinSTS-Retrieval benchmark for Minangkabau."""
    
    def __init__(self, benchmark_path: str = "data/processed/benchmark.json"):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def evaluate_retrieval(self, model, split: str = "monolingual",
                           batch_size: int = 128, model_id: str = "") -> dict:
        """Evaluate retrieval: Recall@k, MRR@10, nDCG@10."""
        if split == "monolingual":
            queries = self.data["retrieval"]["queries_monolingual"]
        elif split == "cross_en":
            queries = self.data["retrieval"]["queries_cross_en"]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        corpus = self.data["retrieval"]["corpus"]
        
        # Encode corpus as documents
        corpus_emb = encode_with_model(model, corpus, batch_size, 
                                       task="retrieval", role="document", model_id=model_id)
        
        # Encode queries
        query_texts = [q["query"] for q in queries]
        query_emb = encode_with_model(model, query_texts, batch_size,
                                      task="retrieval", role="query", model_id=model_id)
        
        sim_matrix = cosine_similarity(query_emb, corpus_emb)
        return self._compute_retrieval_metrics(sim_matrix, queries)
    
    def _compute_retrieval_metrics(self, sim_matrix: np.ndarray, queries: list[dict]) -> dict:
        recall_at_1, recall_at_10, mrr_at_10, ndcg_at_10 = [], [], [], []
        
        for i, query in enumerate(queries):
            relevant = set(query["relevant_doc_ids"])
            if not relevant:
                continue
            
            sorted_idx = np.argsort(sim_matrix[i])[::-1]
            
            recall_at_1.append(1.0 if sorted_idx[0] in relevant else 0.0)
            recall_at_10.append(len(set(sorted_idx[:10]) & relevant) / len(relevant))
            
            rr = 0.0
            for rank, idx in enumerate(sorted_idx[:10], 1):
                if idx in relevant:
                    rr = 1.0 / rank
                    break
            mrr_at_10.append(rr)
            
            dcg = sum(1.0 / np.log2(rank + 1) for rank, idx in enumerate(sorted_idx[:10], 1) if idx in relevant)
            idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), 10) + 1))
            ndcg_at_10.append(dcg / idcg if idcg > 0 else 0.0)
        
        raw = {
            "Recall@1": recall_at_1, "Recall@10": recall_at_10,
            "MRR@10": mrr_at_10, "nDCG@10": ndcg_at_10,
        }
        results = {k: np.mean(v) for k, v in raw.items()}
        results["n_queries"] = len(recall_at_1)
        results["confidence_intervals"] = _bootstrap_ci(raw)
        return results
    
    def evaluate_sts(self, model, batch_size: int = 128, model_id: str = "") -> dict:
        """Evaluate STS: Spearman correlation between predicted and gold similarity."""
        sts_pairs = self.data["sts"]
        s1 = [p["sentence1"] for p in sts_pairs]
        s2 = [p["sentence2"] for p in sts_pairs]
        gold = np.array([p["score"] for p in sts_pairs])
        
        emb1, emb2 = encode_sts_with_model(model, s1, s2, batch_size, model_id)
        
        pred = np.array([np.dot(emb1[i], emb2[i]) for i in range(len(emb1))])
        pred = (pred + 1) * 2.5  # rescale [-1,1] → [0,5]
        
        spearman, pvalue = spearmanr(pred, gold)
        
        by_type = {}
        for i, pair in enumerate(sts_pairs):
            t = pair["type"]
            by_type.setdefault(t, {"pred": [], "gold": []})
            by_type[t]["pred"].append(pred[i])
            by_type[t]["gold"].append(gold[i])
        
        type_corr = {}
        for t, d in by_type.items():
            if len(d["pred"]) > 2:
                sp, pv = spearmanr(d["pred"], d["gold"])
                type_corr[t] = {"spearman": sp, "pvalue": pv, "n": len(d["pred"])}
        
        # Bootstrap CI
        rng = np.random.default_rng(42)
        boot = [spearmanr(pred[rng.choice(len(pred), len(pred), replace=True)],
                          gold[rng.choice(len(gold), len(gold), replace=True)])[0]
                for _ in range(1000)]
        
        return {
            "Spearman": spearman, "pvalue": pvalue,
            "n_pairs": len(sts_pairs), "by_type": type_corr,
            "Spearman_CI95": (np.percentile(boot, 2.5), np.percentile(boot, 97.5)),
        }
    
    def evaluate_cross_lingual(self, model, batch_size: int = 128, model_id: str = "") -> dict:
        """Evaluate cross-lingual bitext retrieval: min↔en, min↔id."""
        results = {}
        
        for lang_pair in ["min_en", "min_id"]:
            pairs = self.data["cross_lingual"][lang_pair]
            if not pairs:
                continue
            
            langs = lang_pair.split("_")
            t1 = [p[langs[0]] for p in pairs]
            t2 = [p[langs[1]] for p in pairs]
            
            emb1, emb2 = encode_sts_with_model(model, t1, t2, batch_size, model_id)
            sim = cosine_similarity(emb1, emb2)
            
            acc1 = sum(1 for i in range(len(pairs)) if np.argmax(sim[i]) == i) / len(pairs)
            mrr = sum(
                next((1.0/rank for rank, idx in enumerate(np.argsort(sim[i])[::-1], 1) if idx == i), 0.0)
                for i in range(len(pairs))
            ) / len(pairs)
            r5 = sum(1 for i in range(len(pairs)) if i in set(np.argsort(sim[i])[::-1][:5])) / len(pairs)
            r10 = sum(1 for i in range(len(pairs)) if i in set(np.argsort(sim[i])[::-1][:10])) / len(pairs)
            
            results[lang_pair] = {
                "Accuracy@1": acc1, "MRR": mrr,
                "Recall@5": r5, "Recall@10": r10,
                "n_pairs": len(pairs),
            }
        
        return results
    
    def evaluate_codeswitch(self, model, batch_size: int = 128, model_id: str = "") -> dict:
        """Evaluate code-switching robustness."""
        cs = self.data["codeswitch"]
        if not cs:
            return {"n_examples": 0}
        
        anchors = [e["text_a"] for e in cs]
        positives = [e["text_b"] for e in cs]
        scores = np.array([e["score"] for e in cs])
        
        emb_a, emb_b = encode_sts_with_model(model, anchors, positives, batch_size, model_id)
        cos_sims = np.array([np.dot(emb_a[i], emb_b[i]) for i in range(len(emb_a))])
        
        pred_scaled = (cos_sims + 1) * 2.5
        gold = scores * 5
        sp, pv = (spearmanr(pred_scaled, gold) if len(set(gold)) > 1 else (0.0, 1.0))
        
        return {
            "avg_cosine_similarity": float(np.mean(cos_sims)),
            "Spearman": sp, "pvalue": pv,
            "n_examples": len(cs),
        }
    
    def evaluate_all(self, model, model_id: str = "", batch_size: int = 128) -> dict:
        """Run full benchmark."""
        logger.info(f"=== MinSTS-Retrieval Full Evaluation: {model_id} ===")
        
        results = {"model_name": model_id}
        
        logger.info("  Retrieval (monolingual)...")
        results["retrieval_monolingual"] = self.evaluate_retrieval(model, "monolingual", batch_size, model_id)
        
        logger.info("  Retrieval (cross en→min)...")
        results["retrieval_cross_en"] = self.evaluate_retrieval(model, "cross_en", batch_size, model_id)
        
        logger.info("  STS...")
        results["sts"] = self.evaluate_sts(model, batch_size, model_id)
        
        logger.info("  Cross-lingual...")
        results["cross_lingual"] = self.evaluate_cross_lingual(model, batch_size, model_id)
        
        logger.info("  Code-switching...")
        results["codeswitch"] = self.evaluate_codeswitch(model, batch_size, model_id)
        
        return results
    
    @staticmethod
    def print_results(results: dict):
        print("\n" + "=" * 70)
        print(f"MinSTS-Retrieval Results: {results.get('model_name', 'Unknown')}")
        print("=" * 70)
        
        for key, label in [("retrieval_monolingual", "Monolingual Retrieval (min→min)"),
                           ("retrieval_cross_en", "Cross-lingual Retrieval (en→min)")]:
            if key in results:
                r = results[key]
                print(f"\n{label}:")
                for m in ["Recall@1", "Recall@10", "MRR@10", "nDCG@10"]:
                    if m in r:
                        val = r[m]
                        ci = r.get("confidence_intervals", {}).get(m, {}).get("CI_95")
                        if ci and isinstance(ci[0], (int, float)):
                            print(f"  {m}: {val:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
                        else:
                            print(f"  {m}: {val:.4f}")
        
        if "sts" in results:
            s = results["sts"]
            print(f"\nSemantic Similarity (STS):")
            print(f"  Spearman: {s['Spearman']:.4f}")
            if "Spearman_CI95" in s:
                ci = s["Spearman_CI95"]
                print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            for t, v in s.get("by_type", {}).items():
                print(f"    {t}: ρ={v['spearman']:.4f} (n={v['n']})")
        
        if "cross_lingual" in results:
            print(f"\nCross-lingual Bitext:")
            for pair, v in results["cross_lingual"].items():
                print(f"  {pair}: Acc@1={v['Accuracy@1']:.4f}, MRR={v['MRR']:.4f}, R@10={v['Recall@10']:.4f}")
        
        if "codeswitch" in results and results["codeswitch"].get("n_examples", 0) > 0:
            cs = results["codeswitch"]
            print(f"\nCode-switching Robustness:")
            print(f"  Avg cosine sim: {cs.get('avg_cosine_similarity', 0):.4f}")
            print(f"  Spearman: {cs.get('Spearman', 0):.4f}")
        
        print("=" * 70 + "\n")


def _bootstrap_ci(metric_values: dict, n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(42)
    cis = {}
    for name, values in metric_values.items():
        values = np.array(values)
        boot = [np.mean(values[rng.choice(len(values), len(values), replace=True)]) for _ in range(n_bootstrap)]
        alpha = (1 - ci) / 2
        cis[name] = {
            "mean": float(np.mean(values)),
            "CI_95": [float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))],
        }
    return cis


def _convert_for_json(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(x) for x in obj]
    return obj


def main():
    import argparse
    from sentence_transformers import SentenceTransformer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="data/processed/benchmark.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    logger.info(f"Loading model: {args.model}")
    
    # Model-specific loading
    is_jina = "jina" in args.model.lower()
    
    if is_jina:
        model = SentenceTransformer(
            args.model,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )
    else:
        model = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code)
    
    bench = MinSTSBenchmark(args.benchmark)
    results = bench.evaluate_all(model, model_id=args.model, batch_size=args.batch_size)
    MinSTSBenchmark.print_results(results)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(_convert_for_json(results), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
