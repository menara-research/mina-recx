#!/usr/bin/env python3
"""
Contrastive fine-tuning of embedding models for Minangkabau.

Supports: Jina v5, multilingual-e5, LazarusNLP/NusaBERT, ModernBERT
Full fine-tuning, no LoRA.
Uses MultipleNegativesRankingLoss (InfoNCE) with in-batch negatives.
"""

import json
import logging
import math
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "jinaai/jina-embeddings-v5-text-nano-retrieval": {
        "trust_remote_code": True,
        "max_seq_length": 512,
        "is_jina": True,
    },
    "intfloat/multilingual-e5-small": {
        "trust_remote_code": False,
        "max_seq_length": 512,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
    },
    "LazarusNLP/all-indo-e5-small-v4": {
        "trust_remote_code": False,
        "max_seq_length": 512,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
    },
    "LazarusNLP/all-NusaBERT-base-v4": {
        "trust_remote_code": False,
        "max_seq_length": 512,
    },
    "answerdotai/ModernBERT-base": {
        "trust_remote_code": True,
        "max_seq_length": 512,
        "is_mlm": True,
    },
}


def load_model(model_id: str, output_dir: str) -> SentenceTransformer:
    """Load model with model-specific configuration."""
    cfg = MODEL_CONFIGS.get(model_id, {})
    is_jina = "jina" in model_id.lower()
    trust = is_jina or cfg.get("trust_remote_code", False)

    logger.info(f"Loading model: {model_id}")

    if cfg.get("is_mlm"):
        from sentence_transformers import models
        transformer = models.Transformer(model_id, trust_remote_code=trust)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        model = SentenceTransformer(modules=[transformer, pooling])
    elif is_jina:
        model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )
    else:
        model = SentenceTransformer(model_id, trust_remote_code=trust)
    
    max_seq = cfg.get("max_seq_length", 512)
    model.max_seq_length = max_seq
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total_params/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
    
    return model


def build_input_examples(data_path: str, model_id: str = "") -> list:
    """Build InputExample objects from training data."""
    cfg = MODEL_CONFIGS.get(model_id, {})
    prefix_q = cfg.get("prefix_query", "")
    prefix_p = cfg.get("prefix_passage", "")
    
    with open(data_path, "r", encoding="utf-8") as f:
        examples_raw = json.load(f)
    
    examples = []
    for ex in examples_raw:
        anchor = prefix_q + ex["anchor"]
        positive = prefix_p + ex["positive"]
        examples.append(InputExample(texts=[anchor, positive], label=1.0))
    
    logger.info(f"Built {len(examples)} InputExamples")
    return examples


def train(
    model_id: str = "jinaai/jina-embeddings-v5-text-nano-retrieval",
    data_path: str = "data/processed/training_with_negatives.json",
    output_dir: str = "models/minang-embedder",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    temperature: float = 0.05,
    seed: int = 42,
):
    """Full fine-tuning with MultipleNegativesRankingLoss."""
    
    torch.manual_seed(seed)
    torch.backends.cuda.enable_cudnn_sdp(False)

    output_path = Path(output_dir) / model_id.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    model = load_model(model_id, str(output_path))
    examples = build_input_examples(data_path, model_id)
    train_dataloader = NoDuplicatesDataLoader(examples, batch_size=batch_size)

    scale = 1.0 / temperature
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=scale)

    warmup_steps = math.ceil(len(train_dataloader) * epochs * warmup_ratio)
    
    logger.info(f"Training: {model_id}")
    logger.info(f"  Examples: {len(examples)}, Epochs: {epochs}, Batch: {batch_size}")
    logger.info(f"  LR: {learning_rate}, Temperature: {temperature}, Warmup: {warmup_steps}")
    logger.info(f"  Output: {output_path}")
    
    # Train (no AMP — model is already bfloat16, AMP scaler breaks with bf16)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=True,
        checkpoint_path=str(output_path / "checkpoints"),
        checkpoint_save_steps=len(train_dataloader),
    )
    
    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="jinaai/jina-embeddings-v5-text-nano-retrieval",
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data", type=str, default="data/processed/training_with_negatives.json")
    parser.add_argument("--output-dir", type=str, default="models/minang-embedder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.backends.cuda.enable_cudnn_sdp(False)
    
    train(
        model_id=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        seed=args.seed,
    )
