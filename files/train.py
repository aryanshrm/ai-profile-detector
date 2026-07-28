"""
train.py — Fine-tune google/vit-base-patch16-224 on local AI vs Real images.
Day 2: Fine-tuning support for Fake AI Profile Detector.

Labels
------
  0  →  AI-generated   (test_images/ai/)
  1  →  Real / human   (test_images/real/)

Output
------
  ./fine_tuned_vit/                      — HuggingFace model + processor artefacts
  ./fine_tuned_vit/training_curves.png   — loss + accuracy plot
  ./fine_tuned_vit/eval_results.json     — final val accuracy + metrics

Usage
-----
  python train.py                          # default args
  python train.py --epochs 3 --batch_size 16
  python train.py --ai_dir path/to/ai --real_dir path/to/real
"""

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate


# ──────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ──────────────────────────────────────────────
# 1.  Dataset helpers
# ──────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def collect_image_paths(root: Path, label: int) -> list:
    """Return [(path_str, label), …] for every supported image under *root*."""
    items = []
    for p in root.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXTS:
            items.append((str(p), label))
    return items


class AIRealDataset(Dataset):
    """Torch Dataset: PIL image → ViTImageProcessor tensors + label."""

    def __init__(self, items: list, processor: ViTImageProcessor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Could not open {path}: {exc}. Using blank image.")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)   # (3, 224, 224)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ──────────────────────────────────────────────
# 2.  Metrics
# ──────────────────────────────────────────────

_accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return _accuracy_metric.compute(predictions=preds, references=labels)


# ──────────────────────────────────────────────
# 3.  Plot helper
# ──────────────────────────────────────────────

def plot_training_curves(log_history: list, output_path: str):
    """Parse Trainer log_history and save a loss + accuracy PNG."""
    train_steps, train_losses = [], []
    val_epochs, val_losses, val_accs = [], [], []

    for entry in log_history:
        if "eval_loss" in entry:
            val_epochs.append(entry.get("epoch", len(val_epochs) + 1))
            val_losses.append(entry["eval_loss"])
            val_accs.append(entry.get("eval_accuracy", 0.0) * 100)
        elif "loss" in entry:
            train_steps.append(entry.get("step", len(train_steps) + 1))
            train_losses.append(entry["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — Loss subplot —
    ax = axes[0]
    if train_losses:
        ax.plot(train_steps, train_losses, label="Train loss",
                color="#4C72B0", linewidth=1.5, alpha=0.8)
    if val_losses:
        ax.plot(val_epochs, val_losses, label="Val loss",
                color="#DD8452", linewidth=2, marker="o", markersize=5)
    ax.set_title("Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step / Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # — Accuracy subplot —
    ax = axes[1]
    if val_accs:
        ax.plot(val_epochs, val_accs, label="Val accuracy",
                color="#55A868", linewidth=2, marker="s", markersize=5)
        ax.axhline(y=85, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="85% target")
    ax.set_title("Validation Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("ViT Fine-tuning — AI vs Real Image Detector", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train] Training curves saved -> {output_path}")



# ──────────────────────────────────────────────
# 4.  Arg parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT for AI vs Real image classification"
    )
    parser.add_argument("--ai_dir",        default="data/dataset/ai",
                        help="Folder of AI-generated images  [label 0]")
    parser.add_argument("--real_dir",      default="data/dataset/real",
                        help="Folder of real/human images    [label 1]")

    parser.add_argument("--output_dir",    default="./fine_tuned_vit",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--base_model",    default="google/vit-base-patch16-224",
                        help="HuggingFace base model ID")
    parser.add_argument("--epochs",        type=int,   default=5,
                        help="Training epochs (3–5 recommended)")
    parser.add_argument("--batch_size",    type=int,   default=8,
                        help="Per-device train/eval batch size (8–16)")
    parser.add_argument("--lr",            type=float, default=5e-5,
                        help="Peak learning rate")
    parser.add_argument("--val_split",     type=float, default=0.20,
                        help="Fraction held out for validation")
    parser.add_argument("--warmup_ratio",  type=float, default=0.10,
                        help="Fraction of total steps used for LR warm-up")
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Stop if val accuracy doesn't improve for N evals")
    return parser.parse_args()


# ──────────────────────────────────────────────
# 5.  Main training routine
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device + precision ────────────────────
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    fp16    = device == "cuda"
    bf16    = False   # safer default; flip to True on Ampere+ GPUs if needed
    print(f"[train] Device : {device}  |  FP16 : {fp16}")
    print(f"[train] Base   : {args.base_model}")
    print(f"[train] Output : {args.output_dir}")

    # ── Collect images ────────────────────────
    ai_dir   = Path(args.ai_dir)
    real_dir = Path(args.real_dir)

    if not ai_dir.exists():
        raise FileNotFoundError(f"AI image directory not found: {ai_dir}")
    if not real_dir.exists():
        raise FileNotFoundError(f"Real image directory not found: {real_dir}")

    ai_items   = collect_image_paths(ai_dir,   label=0)
    real_items = collect_image_paths(real_dir, label=1)

    if not ai_items:
        raise ValueError(f"No supported images found in {ai_dir}")
    if not real_items:
        raise ValueError(f"No supported images found in {real_dir}")

    print(f"[train] AI images   : {len(ai_items)}")
    print(f"[train] Real images : {len(real_items)}")

    # ── Train / val split ─────────────────────
    # Confirmed feedback samples are deliberately kept in the training set.
    # Holding one of them out at random makes it impossible for fine-tuning to
    # learn from the correction supplied by a user.
    all_items = ai_items + real_items
    feedback_items = [
        item for item in all_items
        if Path(item[0]).stem.lower().startswith("feedback_")
    ]
    split_items = [item for item in all_items if item not in feedback_items]
    random.shuffle(split_items)

    n_val = max(1, int(len(split_items) * args.val_split))
    val_items = split_items[:n_val]
    train_items = split_items[n_val:] + feedback_items

    print(f"[train] Train samples : {len(train_items)}")
    print(f"[train] Val samples   : {len(val_items)}")

    # ── Processor ─────────────────────────────
    processor = ViTImageProcessor.from_pretrained(args.base_model)

    train_dataset = AIRealDataset(train_items, processor)
    val_dataset   = AIRealDataset(val_items,   processor)

    # ── Model ─────────────────────────────────
    id2label = {0: "artificial", 1: "real"}
    label2id = {"artificial": 0, "real": 1}

    model = ViTForImageClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # replaces the classification head
    )

    # ── Training arguments ────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=fp16,
        bf16=bf16,
        # Evaluation & checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        report_to="none",               # no wandb / tensorboard dependency
        # Reproducibility
        seed=SEED,
        data_seed=SEED,
        # Misc
        remove_unused_columns=False,    # keep pixel_values + labels
        save_total_limit=2,
        dataloader_num_workers=0,       # 0 = safe on all platforms
    )

    # ── Trainer ───────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    # ── Train ─────────────────────────────────
    print("[train] Starting fine-tuning …")
    train_result = trainer.train()
    print("[train] Training complete.")

    # ── Save model + processor ────────────────
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[train] Model saved -> {args.output_dir}")

    # ── Final evaluation ──────────────────────
    print("[train] Running final evaluation ...")
    eval_results = trainer.evaluate()

    final_acc = eval_results.get("eval_accuracy", 0.0)
    print(f"\n{'=' * 45}")
    print(f"  Final Val Accuracy : {final_acc * 100:.2f}%")
    print(f"  Final Val Loss     : {eval_results.get('eval_loss', 0.0):.4f}")
    print(f"{'=' * 45}\n")

    # Save eval results as JSON (app.py reads this for "test accuracy" display)
    eval_path = os.path.join(args.output_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_accuracy": final_acc,
                "eval_loss": eval_results.get("eval_loss", 0.0),
                "train_samples": len(train_items),
                "val_samples": len(val_items),
                "epochs": args.epochs,
                "base_model": args.base_model,
            },
            f,
            indent=2,
        )
    print(f"[train] Eval results saved -> {eval_path}")

    # ── Plot training curves ──────────────────
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(trainer.state.log_history, plot_path)


    return final_acc


if __name__ == "__main__":
    main()
