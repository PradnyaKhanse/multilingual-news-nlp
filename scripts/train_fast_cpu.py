"""
Fast CPU Fine-Tuning & Evaluation Script for News Classifier
============================================================
Fine-tunes XLM-RoBERTa top layers on CPU using balanced dataset subset,
logging loss at every 5 steps, saving the trained model to models/classifier/best_model,
and printing trainer.state.log_history.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG
from src.preprocess import clean_hindi_text
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def run_fast_cpu_training():
    out_dir = CONFIG.models_dir / "classifier" / "best_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FAST CPU CLASSIFIER FINE-TUNING (XLM-RoBERTa)")
    print(f"Target Output Directory: {out_dir.resolve()}")
    print("=" * 70)

    train_path = CONFIG.processed_data_dir / "train_classify.csv"
    val_path = CONFIG.processed_data_dir / "val_classify.csv"

    t_df = pd.read_csv(train_path)
    v_df = pd.read_csv(val_path)

    # Subsample 100 per class (500 total) for fast CPU training speed
    sub_dfs = []
    for cat in ["Politics", "Sports", "Business", "Entertainment", "Technology"]:
        cat_df = t_df[t_df["category"] == cat]
        sub_dfs.append(cat_df.sample(n=min(100, len(cat_df)), random_state=42))
    train_sub = pd.concat(sub_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Subsampled Training Dataset: {len(train_sub)} samples (100 per class)")
    print(f"Validation Dataset:          {len(v_df)} samples (100 per class)")

    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(examples):
        return tokenizer(
            examples["clean_text"],
            truncation=True,
            max_length=64,
            padding=False
        )

    train_ds = Dataset.from_pandas(train_sub[["clean_text", "label_id"]].rename(columns={"label_id": "label"}))
    val_ds = Dataset.from_pandas(v_df[["clean_text", "label_id"]].rename(columns={"label_id": "label"}))

    train_tokenized = train_ds.map(tokenize_func, batched=True)
    val_tokenized = val_ds.map(tokenize_func, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,
        id2label=CONFIG.id2label,
        label2id=CONFIG.label2id
    )

    # Freeze bottom 10 transformer encoder layers for 5x faster CPU training
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        for layer in model.roberta.encoder.layer[:10]:
            for param in layer.parameters():
                param.requires_grad = False
        print("[Optimization] Freezing bottom 10 XLM-RoBERTa encoder layers for high-speed CPU gradient update...")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p, r, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        return {
            "accuracy": acc,
            "precision_macro": p,
            "recall_macro": r,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=5,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n[Training] Starting gradient descent fine-tuning...")
    train_result = trainer.train()

    print("\n[Training Complete] Saving trained weights and tokenizer...")
    trainer.save_model(str(out_dir.resolve()))
    tokenizer.save_pretrained(str(out_dir.resolve()))

    print("\n" + "=" * 70)
    print("TRAINER LOG HISTORY (Loss Drop & Metrics per Step/Epoch):")
    print("=" * 70)
    for entry in trainer.state.log_history:
        print(entry)

    print(f"\n✅ Fine-tuning completed! Trained model saved at: {out_dir.resolve()}")


if __name__ == "__main__":
    run_fast_cpu_training()
