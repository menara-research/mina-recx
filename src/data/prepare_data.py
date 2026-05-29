#!/usr/bin/env python3
"""
Data preparation for Minangkabau embedding training.
Downloads NusaX bitext, sentiment, and builds training pairs.
Uses parallel downloads and disk caching.
"""

import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
PROCESSED_DIR = Path("data/processed")


def _cached_json(path: Path, compute_fn):
    """Load from cache if exists, else compute and save."""
    if path.exists():
        logger.info(f"Loading cached: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Cached: {path}")
    return result


def load_all_datasets(cache_dir: str = "data/cache") -> dict:
    """Load all NusaX datasets with HF cache (parallel-friendly)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading NusaX bitext (eng-min, eng-ind)...")
    eng_min = load_dataset("mteb/NusaXBitextMining", "eng-min", cache_dir=cache_dir)
    eng_ind = load_dataset("mteb/NusaXBitextMining", "eng-ind", cache_dir=cache_dir)
    
    logger.info("Loading NusaX sentiment (min, ind, eng)...")
    senti = {}
    for lang in ["min", "ind", "eng"]:
        senti[lang] = load_dataset("mteb/nusa_x_senti", lang, cache_dir=cache_dir)
    
    return {"eng_min": eng_min, "eng_ind": eng_ind, "senti": senti}


def build_training_pairs(data: dict, seed: int = 42) -> list[dict]:
    """
    Build training pairs from parallel and sentiment data.
    Types: parallel_translation, same_sentiment, different_sentiment,
           cross_lingual_sentiment, bridge_translation, code_switch
    """
    random.seed(seed)
    pairs = []
    
    eng_min = data["eng_min"]
    eng_ind = data["eng_ind"]
    senti = data["senti"]
    
    # 1. Parallel translation pairs (eng↔min) — strongest signal
    for row in eng_min["train"]:
        pairs.append({
            "type": "parallel_translation",
            "text_a": row["sentence2"],  # min
            "text_b": row["sentence1"],  # eng
            "lang_a": "min", "lang_b": "eng",
            "score": 1.0,
        })
    
    # 2. Same-sentiment pairs within minangkabau
    for split in ["train", "validation"]:
        by_label = {0: [], 1: [], 2: []}
        for row in senti["min"][split]:
            by_label[row["label"]].append(row["text"])
        
        for label, texts in by_label.items():
            for i in range(len(texts)):
                for j in range(i + 1, min(i + 5, len(texts))):
                    pairs.append({
                        "type": "same_sentiment",
                        "text_a": texts[i], "text_b": texts[j],
                        "lang_a": "min", "lang_b": "min",
                        "score": 0.8 if label != 1 else 0.6,
                    })
        
        # Hard negatives: positive vs negative sentiment
        if by_label[0] and by_label[2]:
            for i in range(min(50, len(by_label[0]))):
                for j in range(min(2, len(by_label[2]))):
                    pairs.append({
                        "type": "different_sentiment",
                        "text_a": by_label[0][i], "text_b": by_label[2][j],
                        "lang_a": "min", "lang_b": "min",
                        "score": 0.1,
                    })
    
    # 3. Cross-lingual sentiment alignment (min↔eng, min↔id)
    for other_lang in ["eng", "ind"]:
        by_label_min = {0: [], 1: [], 2: []}
        by_label_other = {0: [], 1: [], 2: []}
        for row in senti["min"]["train"]:
            by_label_min[row["label"]].append(row["text"])
        for row in senti[other_lang]["train"]:
            by_label_other[row["label"]].append(row["text"])
        
        for label in [0, 1, 2]:
            n = min(len(by_label_min[label]), len(by_label_other[label]), 30)
            for i in range(n):
                pairs.append({
                    "type": "cross_lingual_sentiment",
                    "text_a": by_label_min[label][i],
                    "text_b": by_label_other[label][i],
                    "lang_a": "min", "lang_b": other_lang,
                    "score": 0.7 if label != 1 else 0.5,
                })
    
    # 4. eng↔id bridge translations
    for row in eng_ind["train"]:
        pairs.append({
            "type": "bridge_translation",
            "text_a": row["sentence2"],  # ind
            "text_b": row["sentence1"],  # eng
            "lang_a": "ind", "lang_b": "eng",
            "score": 1.0,
        })
    
    # 5. Code-switching synthetic examples
    pairs.extend(_generate_code_switch(senti, n=200, seed=seed))
    
    random.shuffle(pairs)
    
    type_counts = {}
    for p in pairs:
        t = p["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"Built {len(pairs)} training pairs:")
    for t, c in sorted(type_counts.items()):
        logger.info(f"  {t}: {c}")
    
    return pairs


def _generate_code_switch(senti: dict, n: int = 200, seed: int = 42) -> list[dict]:
    """Generate synthetic code-switching examples (min/id/en mix)."""
    random.seed(seed)
    pairs = []
    
    min_by_id = {r["id"]: r for r in senti["min"]["train"]}
    ind_by_id = {r["id"]: r for r in senti["ind"]["train"]}
    eng_by_id = {r["id"]: r for r in senti["eng"]["train"]}
    common = sorted(set(min_by_id) & set(ind_by_id) & set(eng_by_id))
    
    for sid in random.sample(common, min(n, len(common))):
        min_text = min_by_id[sid]["text"]
        eng_text = eng_by_id[sid]["text"]
        ind_text = ind_by_id[sid]["text"]
        
        min_words = min_text.split()
        ind_words = ind_text.split()
        
        if len(min_words) > 3:
            n_replace = random.randint(1, max(1, len(min_words) // 3))
            indices = random.sample(range(len(min_words)), min(n_replace, len(min_words)))
            cs_words = list(min_words)
            for i in indices:
                if i < len(ind_words):
                    cs_words[i] = ind_words[i]
            cs_text = " ".join(cs_words)
            
            pairs.append({
                "type": "code_switch_min_id",
                "text_a": cs_text, "text_b": min_text,
                "lang_a": "min_id", "lang_b": "min",
                "score": 0.85,
            })
            pairs.append({
                "type": "code_switch_min_en",
                "text_a": cs_text, "text_b": eng_text,
                "lang_a": "min_id", "lang_b": "eng",
                "score": 0.75,
            })
    
    return pairs


def build_benchmark(data: dict, seed: int = 42) -> dict:
    """
    Build MinSTS-Retrieval benchmark:
    1. Retrieval (monolingual min→min, cross en→min)
    2. STS (sentence pairs with similarity scores)
    3. Cross-lingual bitext (min↔en, min↔id)
    4. Code-switching robustness
    """
    random.seed(seed)
    np.random.seed(seed)
    
    senti = data["senti"]
    eng_min = data["eng_min"]
    
    benchmark = {}
    
    # --- 1. RETRIEVAL ---
    min_test = list(senti["min"]["test"])
    min_corpus = [r["text"] for r in min_test]
    
    min_queries = []
    for i, row in enumerate(min_test):
        relevant = [j for j, r in enumerate(min_test) if r["label"] == row["label"] and j != i]
        if relevant:
            min_queries.append({
                "query_id": f"min-q-{i}",
                "query": row["text"],
                "relevant_doc_ids": relevant[:10],
            })
    
    # Cross-lingual queries (en→min corpus)
    eng_test = list(senti["eng"]["test"])
    eng_by_id = {r["id"]: r for r in eng_test}
    min_by_id = {r["id"]: r for r in min_test}
    common_en = set(eng_by_id) & set(min_by_id)
    
    cross_queries_en = []
    for sid in common_en:
        eng_row = eng_by_id[sid]
        min_idx = next(i for i, r in enumerate(min_test) if r["id"] == sid)
        same_label = [j for j, r in enumerate(min_test) if r["label"] == eng_row["label"] and j != min_idx]
        if same_label:
            cross_queries_en.append({
                "query_id": f"en-min-q-{sid}",
                "query": eng_row["text"],
                "relevant_doc_ids": same_label[:10],
            })
    
    benchmark["retrieval"] = {
        "corpus": min_corpus,
        "queries_monolingual": min_queries,
        "queries_cross_en": cross_queries_en,
    }
    
    # --- 2. STS ---
    sts_pairs = []
    
    # Parallel = high similarity
    for row in eng_min["train"]:
        sts_pairs.append({
            "sentence1": row["sentence2"], "sentence2": row["sentence1"],
            "score": 5.0, "type": "parallel_translation", "lang_pair": "min-en",
        })
    
    by_label = {0: [], 1: [], 2: []}
    for row in min_test:
        by_label[row["label"]].append(row["text"])
    
    # Same sentiment
    for label, texts in by_label.items():
        n = min(30, len(texts))
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):
                score = 4.0 if label in [0, 2] else 3.0
                sts_pairs.append({
                    "sentence1": texts[i], "sentence2": texts[j],
                    "score": score, "type": "same_sentiment", "lang_pair": "min-min",
                })
    
    # Different sentiment
    for la, lb, sc in [(0, 2, 1.0), (0, 1, 1.5), (1, 2, 1.5)]:
        if by_label[la] and by_label[lb]:
            for i in range(min(20, len(by_label[la]))):
                j = random.randint(0, len(by_label[lb]) - 1)
                sts_pairs.append({
                    "sentence1": by_label[la][i], "sentence2": by_label[lb][j],
                    "score": sc, "type": "different_sentiment", "lang_pair": "min-min",
                })
    
    benchmark["sts"] = sts_pairs
    
    # --- 3. CROSS-LINGUAL ---
    ind_test = list(senti["ind"]["test"])
    ind_by_id = {r["id"]: r for r in ind_test}
    common_min_ind = set(min_by_id) & set(ind_by_id)
    
    cross_lingual = {"min_en": [], "min_id": []}
    for sid in common_en:
        cross_lingual["min_en"].append({
            "min": min_by_id[sid]["text"], "en": eng_by_id[sid]["text"],
            "label": min_by_id[sid]["label"],
        })
    for sid in common_min_ind:
        cross_lingual["min_id"].append({
            "min": min_by_id[sid]["text"], "id": ind_by_id[sid]["text"],
            "label": min_by_id[sid]["label"],
        })
    benchmark["cross_lingual"] = cross_lingual
    
    # --- 4. CODE-SWITCHING ---
    benchmark["codeswitch"] = _generate_code_switch(senti, n=50, seed=seed)
    
    logger.info("=== MinSTS-Retrieval Benchmark Stats ===")
    logger.info(f"  Retrieval: {len(min_queries)} mono, {len(cross_queries_en)} cross-en, {len(min_corpus)} corpus")
    logger.info(f"  STS: {len(sts_pairs)} pairs")
    logger.info(f"  Cross-lingual: min-en={len(cross_lingual['min_en'])}, min-id={len(cross_lingual['min_id'])}")
    logger.info(f"  Code-switching: {len(benchmark['codeswitch'])}")
    
    return benchmark


def mine_hard_negatives_bm25(queries: list[str], corpus: list[str], top_k: int = 10) -> list[list[int]]:
    """Mine hard negatives using BM25."""
    from rank_bm25 import BM25Okapi
    
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    hard_negs = []
    for query in tqdm(queries, desc="BM25 hard negatives"):
        scores = bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        negs = [idx for idx in top_indices if corpus[idx] != query][:top_k]
        hard_negs.append(negs)
    
    return hard_negs


def prepare_training_with_negatives(pairs: list[dict], n_neg: int = 5) -> list[dict]:
    """Prepare training data with BM25 hard negatives for contrastive learning."""
    positive_pairs = [p for p in pairs if p["score"] >= 0.7]
    
    all_texts = list({p["text_a"] for p in positive_pairs} | {p["text_b"] for p in positive_pairs})
    queries = [p["text_a"] for p in positive_pairs]
    
    logger.info(f"Mining hard negatives: {len(queries)} queries, {len(all_texts)} corpus...")
    hard_negs = mine_hard_negatives_bm25(queries, all_texts, top_k=n_neg + 2)
    
    examples = []
    for i, pair in enumerate(tqdm(positive_pairs, desc="Building examples")):
        neg_texts = [all_texts[idx] for idx in hard_negs[i] if all_texts[idx] != pair["text_b"]][:n_neg]
        examples.append({
            "anchor": pair["text_a"],
            "positive": pair["text_b"],
            "negatives": neg_texts,
            "type": pair["type"],
            "lang_pair": f"{pair['lang_a']}-{pair['lang_b']}",
        })
    
    logger.info(f"Prepared {len(examples)} training examples with hard negatives")
    return examples


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger.info("=== Minangkabau Embedding: Data Preparation ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load with HF cache
    data = load_all_datasets()
    
    # Build training pairs (cached)
    pairs = _cached_json(
        PROCESSED_DIR / "training_pairs.json",
        lambda: build_training_pairs(data),
    )
    
    # Build benchmark (cached)
    benchmark = _cached_json(
        PROCESSED_DIR / "benchmark.json",
        lambda: build_benchmark(data),
    )
    
    # Build training with hard negatives (cached)
    training_examples = _cached_json(
        PROCESSED_DIR / "training_with_negatives.json",
        lambda: prepare_training_with_negatives(pairs),
    )
    
    # Also save as HF Dataset for easy loading
    flat = {
        "anchor": [e["anchor"] for e in training_examples],
        "positive": [e["positive"] for e in training_examples],
        "negatives": [e["negatives"] for e in training_examples],
        "type": [e["type"] for e in training_examples],
        "lang_pair": [e["lang_pair"] for e in training_examples],
    }
    ds = Dataset.from_dict(flat)
    ds.save_to_disk(str(PROCESSED_DIR / "train_dataset_hf"))
    
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
