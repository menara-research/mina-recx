"""Smoke tests. Run: uv run python tests/test_pipeline.py"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.evaluate import MinSTSBenchmark, _convert_for_json

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/processed/benchmark.json"


def test_no_leakage():
    bench = json.loads(BENCH.read_text())
    train = json.loads((ROOT / "data/processed/training_pairs.json").read_text())
    train_pairs = {(p["text_a"], p["text_b"]) for p in train}
    for p in bench["sts"]:
        assert (p["sentence1"], p["sentence2"]) not in train_pairs, "STS/training leakage"


def test_retrieval_metrics():
    # 2 queries, relevant doc at index 0 for each, ranked first by similarity.
    sim_matrix = np.array([[0.9, 0.1, 0.2], [0.8, 0.3, 0.1]])
    queries = [{"relevant_doc_ids": [0]}, {"relevant_doc_ids": [0]}]
    r = MinSTSBenchmark._compute_retrieval_metrics(MinSTSBenchmark.__new__(MinSTSBenchmark), sim_matrix, queries)
    assert r["Recall@1"] == 1.0
    assert r["MRR@10"] == 1.0


def test_convert_for_json():
    obj = {
        "f": np.float64(1.5),
        "i": np.int64(3),
        "arr": np.array([1.0, 2.0]),
        "tup": (np.float64(0.1), np.float64(0.2)),
    }
    json.dumps(_convert_for_json(obj))


def test_benchmark_keys():
    bench = json.loads(BENCH.read_text())
    for k in ["retrieval", "sts", "cross_lingual", "codeswitch"]:
        assert k in bench, f"missing key: {k}"


if __name__ == "__main__":
    test_no_leakage()
    test_retrieval_metrics()
    test_convert_for_json()
    test_benchmark_keys()
    print("ok")
