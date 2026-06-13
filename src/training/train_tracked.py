#!/usr/bin/env python3
"""
Training with loss tracking + ablation runs.
Saves per-step loss and final benchmark results for each config.
"""

import json
import logging
import math
from pathlib import Path

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "jinaai/jina-embeddings-v5-text-nano-retrieval": {
        "trust_remote_code": True,
        "max_seq_length": 512,
        "is_jina": True,
    },
}


def load_model(model_id: str) -> SentenceTransformer:
    cfg = MODEL_CONFIGS.get(model_id, {})
    trust = cfg.get("trust_remote_code", False)
    model = SentenceTransformer(model_id, trust_remote_code=trust)
    model.max_seq_length = cfg.get("max_seq_length", 512)
    return model


def build_input_examples(data_path: str, model_id: str = "") -> list:
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
    return examples


def train_with_tracking(
    model_id: str = "jinaai/jina-embeddings-v5-text-nano-retrieval",
    data_path: str = "data/processed/training_with_negatives.json",
    output_dir: str = "models/ablations",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 2e-5,
    temperature: float = 0.05,
    seed: int = 42,
    run_name: str = "default",
):
    """Full fine-tuning with loss tracking."""
    torch.manual_seed(seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    model = load_model(model_id)
    examples = build_input_examples(data_path, model_id)
    train_dataloader = NoDuplicatesDataLoader(examples, batch_size=batch_size)
    
    scale = 1.0 / temperature
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=scale)
    
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)

    # Patch the loss forward to record per-step loss; the HF Trainer used
    # internally by sentence-transformers exposes no per-step loss hook.
    original_forward = train_loss.forward
    
    step_data = {"steps": [], "losses": []}
    call_count = [0]
    
    def tracked_forward(features, labels=None):
        loss = original_forward(features, labels)
        if isinstance(loss, torch.Tensor):
            step_data["steps"].append(call_count[0])
            step_data["losses"].append(loss.item())
            call_count[0] += 1
        return loss
    
    train_loss.forward = tracked_forward
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=True,
        use_amp=False,
    )
    
    train_loss.forward = original_forward

    loss_out = {
        "run_name": run_name,
        "config": {
            "model_id": model_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "seed": seed,
            "total_steps": len(train_dataloader) * epochs,
            "steps_per_epoch": len(train_dataloader),
        },
        "step_losses": step_data,
    }
    with open(output_path / "loss_history.json", "w") as f:
        json.dump(loss_out, f)
    
    # Copy the model's custom modeling file from the HF modules cache so the
    # saved checkpoint reloads later. Path varies by machine/commit, so glob it.
    import shutil
    modules_root = Path.home() / ".cache/huggingface/modules/transformers_modules"
    src = next(modules_root.glob("**/modeling_eurobert.py"), None)
    dst = output_path / "modeling_eurobert.py"
    if src and not dst.exists():
        shutil.copy2(src, dst)
    
    logger.info(f"[{run_name}] Final loss: {step_data['losses'][-1]:.4f}, Steps: {len(step_data['losses'])}")
    return output_path, loss_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.backends.cuda.enable_cudnn_sdp(False)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    
    train_with_tracking(
        run_name=args.run_name,
        epochs=args.epochs,
        temperature=args.temperature,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
