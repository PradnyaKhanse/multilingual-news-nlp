import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG, DEVICE, get_device
from src.preprocess import clean_hindi_text, load_and_clean_classification_data, prepare_classification_splits


# ==============================================================================
# Helper Heuristic Fallback for Zero-Shot/Offline Category Detection
# ==============================================================================

CATEGORY_KEYWORDS = {
    "Politics": [
        "संसद", "सरकार", "विपक्ष", "चुनाव", "पार्टी", "नेता", "मंत्री", "लोकसभा", "राज्यसभा",
        "अध्यक्ष", "कार्यकारिणी", "कूटनीति", "विधेयक", "आयोग", "विधानसभा", "गठबंधन", "राजनीति",
        "parliament", "government", "minister", "election", "opposition", "party", "bjp", "congress", "senate", "assembly",
        "विधानसभा", "अधिवेशन", "विरोधक", "सत्ताधारी", "खासदार", "आमदार", "राजकारण"
    ],
    "Sports": [
        "क्रिकेट", "मैच", "खिलाड़ी", "टीम", "रन", "विकेट", "बल्लेबाज", "गेंदबाज", "टूर्नामेंट",
        "ओलंपिक", "पदक", "चैंपियनशिप", "बैडमिंटन", "मुक्केबाजी", "कप्तान", "फाइनल", "खेल",
        "cricket", "match", "team", "player", "runs", "wicket", "batsman", "bowler", "tournament", "olympics", "medal", "stadium", "victory",
        "सामना", "खेळाडू", "संघ", "धावा", "गोलंदाज", "फलंदाज", "विजय", "स्पर्धा"
    ],
    "Business": [
        "शेयर", "बाजार", "सेंसेक्स", "निफ्टी", "निवेश", "बैंक", "रिजर्व बैंक", "रेपो रेट", "महंगाई",
        "मुद्रास्फीति", "अर्थव्यवस्था", "कंपनी", "उद्योग", "स्टार्टअप", "मुनाफा", "जीडीपी", "व्यापार",
        "sensex", "nifty", "stock", "market", "economy", "investment", "bank", "rbi", "inflation", "profit", "shares", "business", "company",
        "सेन्सेक्स", "शेअर", "अर्थसंकल्प", "गुंतवणूक", "महागाई", "बँक", "व्यवसाय", "उद्योग"
    ],
    "Entertainment": [
        "फिल्म", "अभिनेता", "अभिनेत्री", "सिनेमा", "ट्रेलर", "बॉलीवुड", "गीत", "संगीत", "निर्देशक",
        "ओटीटी", "वेब सीरीज", "कलाकार", "बॉक्स ऑफिस", "थ्रिलर", "ड्रामा", "मनोरंजन",
        "movie", "film", "actor", "actress", "bollywood", "cinema", "trailer", "box office", "director", "song", "series", "entertainment",
        "चित्रपट", "अभिनेता", "अभिनेत्री", "ट्रेलर", "गाणे", "दिग्दर्शक", "मनोरंजन", "कलाकार"
    ],
    "Technology": [
        "तकनीक", "प्रौद्योगिकी", "इसरो", "उपग्रह", "रॉकेट", "सुपरकंप्यूटर", "एआई", "आर्टिफिशियल इंटेलिजेंस",
        "चिपसेट", "सेमीकंडक्टर", "स्मार्टफोन", "इंटरनेट", "5जी", "सॉफ्टवेयर", "नवाचार", "डिजिटल",
        "isro", "satellite", "rocket", "ai", "artificial intelligence", "software", "technology", "tech", "chip", "semiconductor", "internet", "5g",
        "इस्रो", "उपग्रह", "तंत्रज्ञान", "संगणक", "इंटरनेट", "प्रक्षेपका", "डिजिटल"
    ]
}


def heuristic_hindi_classifier(text: str) -> Dict[str, Any]:
    """
    Fast, rule-assisted statistical classifier used as an instant lightweight fallback
    when transformer weights are loading or in ultra-constrained environments.
    """
    cleaned = clean_hindi_text(text)
    scores = {cat: 0.05 for cat in CONFIG.categories}
    
    total_matches = 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in cleaned:
                count = cleaned.count(kw)
                scores[cat] += count * 1.5
                total_matches += count
                
    # Softmax normalization
    score_vals = np.array(list(scores.values()))
    exp_scores = np.exp(score_vals - np.max(score_vals))
    probs = exp_scores / exp_scores.sum()
    
    prob_dict = {cat: float(probs[i]) for i, cat in enumerate(CONFIG.categories)}
    best_cat = max(prob_dict, key=prob_dict.get)
    best_idx = CONFIG.label2id[best_cat]
    
    return {
        "category": best_cat,
        "category_hi": CONFIG.categories_hi[best_idx],
        "confidence": prob_dict[best_cat],
        "probabilities": prob_dict,
        "model_used": "Indic-Keyword-Prior (Heuristic Fallback)"
    }


# ==============================================================================
# Model Architecture & Inference Class
# ==============================================================================

def resolve_classifier_path(model_name_or_path: Optional[str] = None) -> str:
    """
    Robustly resolve local fine-tuned model checkpoint directory or Hub model name.
    
    Searches:
    1. Explicit path (and resolved variants, including deduping /content/PROJ NLP/PROJ NLP)
    2. Standard project locations (PROJECT_DIR / models / classifier / best_model)
    3. Colab standard paths (/content/PROJ NLP/models/classifier/best_model)
    4. Recursive scan under any models/ directory for config.json and weights
    """
    explicit_candidates = []
    if model_name_or_path:
        p = Path(model_name_or_path)
        explicit_candidates.extend([
            p,
            p.resolve() if p.is_absolute() else None,
            PROJECT_DIR / model_name_or_path,
            Path.cwd() / model_name_or_path,
            Path.cwd().parent / model_name_or_path,
        ])
        
        # Handle doubled directory paths e.g. /content/PROJ NLP/PROJ NLP/...
        str_p = str(model_name_or_path).replace("\\", "/")
        if "/PROJ NLP/PROJ NLP" in str_p:
            dedup_p = Path(str_p.replace("/PROJ NLP/PROJ NLP", "/PROJ NLP"))
            explicit_candidates.append(dedup_p)
        if "models/classifier/best_model" in str_p:
            explicit_candidates.extend([
                Path("/content/PROJ NLP/models/classifier/best_model"),
                Path("/content/PROJ NLP/PROJ NLP/models/classifier/best_model"),
                Path("/content/models/classifier/best_model"),
            ])

    standard_candidates = [
        PROJECT_DIR / "models" / "classifier" / "best_model",
        PROJECT_DIR / CONFIG.classification.saved_model_path,
        Path.cwd() / "models" / "classifier" / "best_model",
        Path.cwd() / CONFIG.classification.saved_model_path,
        Path.cwd().parent / "models" / "classifier" / "best_model",
        Path("/content/PROJ NLP/models/classifier/best_model"),
        Path("/content/PROJ NLP/PROJ NLP/models/classifier/best_model"),
        Path("/content/models/classifier/best_model"),
        CONFIG.models_dir / "classifier" / "best_model",
        CONFIG.models_dir / "best_model",
    ]

    all_candidates = [c for c in (explicit_candidates + standard_candidates) if c is not None]
    for cand in all_candidates:
        if cand.exists() and cand.is_dir():
            has_config = (cand / "config.json").exists()
            has_weights = (cand / "model.safetensors").exists() or (cand / "pytorch_model.bin").exists()
            if has_config or has_weights:
                return str(cand.resolve())
            if any(cand.iterdir()):
                return str(cand.resolve())

    # Recursive search across all models/ directories
    search_roots = [
        PROJECT_DIR / "models",
        Path.cwd() / "models",
        Path.cwd().parent / "models",
        Path("/content/PROJ NLP/models"),
        Path("/content/PROJ NLP/PROJ NLP/models"),
        Path("/content/models"),
        CONFIG.models_dir,
    ]
    for s_root in search_roots:
        if s_root.exists() and s_root.is_dir():
            for sub in s_root.rglob("*"):
                if sub.is_dir() and ((sub / "config.json").exists() or (sub / "model.safetensors").exists() or (sub / "pytorch_model.bin").exists()):
                    return str(sub.resolve())

    # If model_name_or_path is an explicit HuggingFace repo (not a failed path lookup)
    if model_name_or_path:
        str_m = str(model_name_or_path)
        if "models" not in str_m and not str_m.startswith(("/", "C:", ".", "\\")):
            return str_m

    return CONFIG.classification.model_name


class HindiNewsClassifier:
    """
    End-to-end wrapper for XLM-RoBERTa / IndicBERT Hindi News Classifier.
    Supports fine-tuning with HF Trainer, metric logging, and production inference.
    """
    
    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name_or_path = resolve_classifier_path(model_name_or_path)
        self.device = get_device()
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load_model(self, force_reload: bool = False):
        """Lazy load tokenizer and model weights with local_files_only for local checkpoints."""
        if self.is_loaded and not force_reload and self.model is not None:
            return
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Re-resolve path in case a new checkpoint directory was written
            resolved = resolve_classifier_path(self.model_name_or_path)
            self.model_name_or_path = resolved
            is_local = Path(resolved).exists() and Path(resolved).is_dir()
            
            if is_local:
                print(f"[Classifier] Loading local fine-tuned checkpoint from: {resolved} on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    resolved,
                    local_files_only=True,
                    num_labels=len(CONFIG.categories),
                    id2label=CONFIG.id2label,
                    label2id=CONFIG.label2id
                )
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                print(f"[Classifier] Successfully loaded fine-tuned checkpoint: {resolved}")
            else:
                print(f"[Classifier] Loading base model from Hub: {resolved} on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(resolved)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    resolved,
                    num_labels=len(CONFIG.categories),
                    id2label=CONFIG.id2label,
                    label2id=CONFIG.label2id
                )
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                print(f"[Classifier] Loaded base model: {resolved}")
        except Exception as e:
            print(f"[Classifier Warning] Could not load transformer model '{self.model_name_or_path}': {e}")
            print("[Classifier] Will use robust Indic Heuristic fallback for classification inference.")
            self.is_loaded = False

    def get_checkpoint_status_info(self) -> Dict[str, str]:
        """Returns clear status metadata about the classification model without blocking UI."""
        resolved = str(self.model_name_or_path)
        is_local = Path(resolved).exists() and Path(resolved).is_dir()
        
        if is_local:
            return {
                "status": "fine_tuned",
                "label": f"Fine-Tuned Checkpoint ({Path(resolved).name})",
                "badge": "🟢 Fine-Tuned Checkpoint",
                "path": resolved
            }
        elif self.model_name_or_path:
            return {
                "status": "base_model",
                "label": f"Base Pretrained Model ({resolved}) — NOT fine-tuned",
                "badge": "🟡 Base Pretrained Model (NOT fine-tuned)",
                "path": resolved
            }
        else:
            return {
                "status": "heuristic_fallback",
                "label": "Heuristic Keyword Fallback",
                "badge": "🔴 Heuristic Keyword Fallback",
                "path": "N/A"
            }

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Classify a Hindi news article into one of 5 categories.
        
        Args:
            text (str): Raw or cleaned Hindi article text.
            top_k (int): Number of top probability classes to return.
            
        Returns:
            dict with category name (EN & HI), confidence score, and all class probabilities.
        """
        cleaned_text = clean_hindi_text(text)
        if not cleaned_text:
            return {
                "category": "Unknown",
                "category_hi": "अज्ञात",
                "confidence": 0.0,
                "probabilities": {cat: 0.0 for cat in CONFIG.categories},
                "model_used": "None (Empty Input)"
            }
            
        # Try transformer inference if loaded
        if not self.is_loaded:
            self.load_model()
            
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                inputs = self.tokenizer(
                    cleaned_text,
                    truncation=True,
                    max_length=CONFIG.classification.max_seq_length,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    
                pred_id = int(np.argmax(probs))
                pred_label = CONFIG.id2label.get(pred_id, "Politics")
                prob_dict = {CONFIG.id2label[i]: float(probs[i]) for i in range(len(CONFIG.categories))}
                
                return {
                    "category": pred_label,
                    "category_hi": CONFIG.categories_hi[pred_id],
                    "confidence": float(probs[pred_id]),
                    "probabilities": prob_dict,
                    "model_used": self.model_name_or_path
                }
            except Exception as e:
                print(f"[Classifier Error during inference]: {e}. Falling back to heuristic.")
                
        # Fallback
        return heuristic_hindi_classifier(cleaned_text)


# Singleton instance for quick inference
_DEFAULT_CLASSIFIER: Optional[HindiNewsClassifier] = None

def get_classifier(model_path: Optional[str] = None, force_reload: bool = False) -> HindiNewsClassifier:
    global _DEFAULT_CLASSIFIER
    target_path = resolve_classifier_path(model_path)
    if _DEFAULT_CLASSIFIER is None or _DEFAULT_CLASSIFIER.model_name_or_path != target_path or force_reload:
        _DEFAULT_CLASSIFIER = HindiNewsClassifier(target_path)
        if force_reload:
            _DEFAULT_CLASSIFIER.load_model(force_reload=True)
    return _DEFAULT_CLASSIFIER


def predict_category(text: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for category prediction."""
    classifier = get_classifier(model_path)
    return classifier.predict(text)


# ==============================================================================
# Model Training & Evaluation Functions
# ==============================================================================

def train_classifier(
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None
) -> str:
    """
    Fine-tune XLM-RoBERTa on Hindi news dataset using Hugging Face Trainer.
    """
    global _DEFAULT_CLASSIFIER
    # Reset singleton to prevent stale model state
    _DEFAULT_CLASSIFIER = None
    
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    model_name = model_name or CONFIG.classification.model_name
    out_dir = Path(output_dir) if output_dir else (PROJECT_DIR / CONFIG.classification.saved_model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = num_epochs or CONFIG.classification.num_epochs
    bs = batch_size or CONFIG.classification.batch_size
    lr = learning_rate or CONFIG.classification.learning_rate
    
    print("=" * 60)
    print(f"Fine-Tuning Classifier: {model_name}")
    print(f"Output Directory: {out_dir.resolve()}")
    print(f"Epochs: {epochs} | Batch Size: {bs} | Learning Rate: {lr}")
    print("=" * 60)
    
    # Load dataset
    if train_csv:
        train_path = Path(train_csv) if Path(train_csv).exists() else PROJECT_DIR / train_csv
    else:
        train_path = CONFIG.processed_data_dir / "train_classify.csv"

    if val_csv:
        val_path = Path(val_csv) if Path(val_csv).exists() else PROJECT_DIR / val_csv
    else:
        val_path = CONFIG.processed_data_dir / "val_classify.csv"
    
    if not train_path.exists() or not val_path.exists():
        print("[Training] Processed splits not found. Generating them now...")
        df = load_and_clean_classification_data()
        t_df, v_df, _ = prepare_classification_splits(df)
    else:
        t_df = pd.read_csv(train_path)
        v_df = pd.read_csv(val_path)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_func(examples):
        return tokenizer(
            examples["clean_text"],
            truncation=True,
            max_length=CONFIG.classification.max_seq_length,
            padding=False
        )
        
    train_ds = Dataset.from_pandas(t_df[["clean_text", "label_id"]].rename(columns={"label_id": "label"}))
    val_ds = Dataset.from_pandas(v_df[["clean_text", "label_id"]].rename(columns={"label_id": "label"}))
    
    train_tokenized = train_ds.map(tokenize_func, batched=True)
    val_tokenized = val_ds.map(tokenize_func, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CONFIG.categories),
        id2label=CONFIG.id2label,
        label2id=CONFIG.label2id
    )
    
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
        
    training_args_dict = dict(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        weight_decay=CONFIG.classification.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none"
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", save_strategy="epoch", **training_args_dict)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", save_strategy="epoch", **training_args_dict)
    
    # Build Trainer kwargs — newer transformers (>=4.46) renamed `tokenizer`
    # to `processing_class`. We try the new name first and fall back gracefully.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # Older transformers (<4.46) still use `tokenizer=`
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    
    print("[Training] Starting training...")
    trainer.train()
    
    print(f"[Training] Saving final best model to {out_dir.resolve()}...")
    trainer.save_model(str(out_dir.resolve()))
    tokenizer.save_pretrained(str(out_dir.resolve()))
    
    # Invalidate singleton to guarantee fresh weights on next inference/eval
    _DEFAULT_CLASSIFIER = None
    
    return str(out_dir.resolve())


def evaluate_classifier_performance(
    test_csv: Optional[str] = None,
    model_path: Optional[str] = None,
    save_plot: bool = True
) -> Dict[str, Any]:
    """
    Evaluate classification performance on test set, print detailed report,
    and generate a Confusion Matrix plot.
    
    Returns:
        Dict with accuracy, precision (macro/weighted), recall (macro/weighted),
        f1 (macro/weighted), classification report, confusion matrix, and plot path.
    """
    import torch
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support
    )
    
    # 1. Resolve test CSV path - scan all candidates and pick the real dataset
    test_path = None
    if test_csv:
        p = Path(test_csv)
        if p.exists():
            test_path = p
        elif (PROJECT_DIR / test_csv).exists():
            test_path = PROJECT_DIR / test_csv
        elif (Path.cwd() / test_csv).exists():
            test_path = Path.cwd() / test_csv
            
    if test_path is None:
        # Check all possible test split locations across Colab, cwd, and project root
        test_candidates = [
            CONFIG.processed_data_dir / "test_classify.csv",
            PROJECT_DIR / "data" / "processed" / "test_classify.csv",
            Path.cwd() / "data" / "processed" / "test_classify.csv",
            Path.cwd().parent / "data" / "processed" / "test_classify.csv",
            Path("/content/PROJ NLP/data/processed/test_classify.csv"),
            Path("/content/PROJ NLP/PROJ NLP/data/processed/test_classify.csv"),
            Path("/content/data/processed/test_classify.csv"),
        ]
        best_cand = None
        max_rows = -1
        for cand in test_candidates:
            if cand.exists():
                try:
                    cand_df = pd.read_csv(cand)
                    if len(cand_df) > max_rows:
                        max_rows = len(cand_df)
                        best_cand = cand
                except Exception:
                    pass
        if best_cand is not None:
            test_path = best_cand

    # If no test split exists or if it is a tiny 3-row sample stub, build from real raw dataset
    need_regenerate = (test_path is None or not test_path.exists())
    if not need_regenerate:
        try:
            curr_df = pd.read_csv(test_path)
            if len(curr_df) <= 15:
                need_regenerate = True
        except Exception:
            need_regenerate = True

    if need_regenerate:
        print(f"[Evaluation] Test split at {test_path} is missing or a small stub. Regenerating splits from full dataset...")
        try:
            df = load_and_clean_classification_data()
            train_df, val_df, test_df = prepare_classification_splits(df)
            saved_paths = save_processed_datasets(train_df, val_df, test_df)
            test_path = saved_paths["test"]
        except Exception as e:
            if test_path and test_path.exists():
                test_df = pd.read_csv(test_path)
            else:
                raise FileNotFoundError(
                    f"[Evaluation Error] Could not find or generate test dataset split: {e}\n"
                    f"Please ensure raw dataset exists in 'data/raw' or pass a valid 'test_csv' path."
                )
    else:
        test_df = pd.read_csv(test_path)
        
    if len(test_df) == 0:
        raise ValueError(f"[Evaluation Error] Test dataset at '{test_path}' is empty (0 rows).")

    # Standardize column names
    if "clean_text" not in test_df.columns:
        if "article" in test_df.columns:
            test_df["clean_text"] = test_df["article"].apply(clean_hindi_text)
        elif "Content" in test_df.columns:
            test_df["clean_text"] = test_df["Content"].apply(clean_hindi_text)
        elif "text" in test_df.columns:
            test_df["clean_text"] = test_df["text"].apply(clean_hindi_text)
        else:
            test_df["clean_text"] = test_df.iloc[:, 0].apply(clean_hindi_text)

    if "category" not in test_df.columns:
        if "Category" in test_df.columns:
            test_df["category"] = test_df["Category"]
        elif "label" in test_df.columns:
            test_df["category"] = test_df["label"]
        elif "label_id" in test_df.columns:
            test_df["category"] = test_df["label_id"].map(CONFIG.id2label)
        else:
            raise KeyError(
                f"[Evaluation Error] Test CSV must contain 'category', 'Category', 'label', or 'label_id' column. "
                f"Found columns: {list(test_df.columns)}"
            )

    # Category alias normalization
    category_alias_map = {
        "politics": "Politics", "राजनीति": "Politics",
        "sports": "Sports", "खेल": "Sports",
        "business": "Business", "व्यापार": "Business", "कारोबार": "Business",
        "entertainment": "Entertainment", "मनोरंजन": "Entertainment",
        "technology": "Technology", "tech": "Technology", "प्रौद्योगिकी": "Technology", "तकनीक": "Technology"
    }
    y_true = [category_alias_map.get(str(c).strip().lower(), str(c).strip()) for c in test_df["category"].tolist()]

    # 2. Load model with forced reload to ensure fine-tuned checkpoint is active
    classifier = get_classifier(model_path, force_reload=True)
    
    # Explicitly check checkpoint type and print
    is_checkpoint = Path(classifier.model_name_or_path).exists() and Path(classifier.model_name_or_path).is_dir()
    checkpoint_type = "Fine-Tuned Checkpoint (Local Disk)" if is_checkpoint else "Base Pretrained Hub Model / Fallback"
    
    print("=" * 65)
    print("CLASSIFICATION MODEL EVALUATION")
    print("=" * 65)
    print(f"Loaded Model Path:  {classifier.model_name_or_path}")
    print(f"Model Type:         {checkpoint_type}")
    print(f"Execution Device:   {classifier.device}")
    print(f"Test Split Path:    {test_path}")
    print(f"Total Test Samples: {len(test_df)} articles")
    print("=" * 65)
    
    # 3. High-speed batched inference if PyTorch model is active
    y_pred = []
    if classifier.is_loaded and classifier.model is not None and classifier.tokenizer is not None:
        texts = [clean_hindi_text(t) for t in test_df["clean_text"].tolist()]
        # Subsample to 200 items for CPU evaluation speed if on CPU
        if str(classifier.device) == "cpu" and len(texts) > 200:
            texts = texts[:200]
            y_true = y_true[:200]
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = classifier.tokenizer(
                batch_texts,
                truncation=True,
                max_length=CONFIG.classification.max_seq_length,
                padding=True,
                return_tensors="pt"
            ).to(classifier.device)
            with torch.no_grad():
                outputs = classifier.model(**inputs)
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            for pid in pred_ids:
                y_pred.append(CONFIG.id2label.get(int(pid), "Politics"))
    else:
        # Fallback to row-by-row predict
        for _, row in test_df.iterrows():
            pred_res = classifier.predict(row["clean_text"])
            y_pred.append(pred_res["category"])

    categories = CONFIG.categories
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    report_str = classification_report(y_true, y_pred, labels=categories, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    print("\nCLASSIFICATION EVALUATION METRICS:")
    print(f"  • Overall Accuracy:    {acc * 100:.2f}%")
    print(f"  • Precision (Macro):   {float(p_macro):.4f} ({float(p_macro)*100:.2f}%)")
    print(f"  • Recall (Macro):      {float(r_macro):.4f} ({float(r_macro)*100:.2f}%)")
    print(f"  • F1-Score (Macro):    {float(f1_macro):.4f}")
    print(f"  • Precision (Weighted):{float(p_wt):.4f} ({float(p_wt)*100:.2f}%)")
    print(f"  • Recall (Weighted):   {float(r_wt):.4f} ({float(r_wt)*100:.2f}%)")
    print(f"  • F1-Score (Weighted): {float(f1_wt):.4f}")
    print("\nDetailed Per-Class Metrics:")
    print(report_str)
    
    # Generate Confusion Matrix Plot
    plot_file = None
    if save_plot:
        plot_dir = CONFIG.outputs_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_file = plot_dir / "classification_confusion_matrix.png"
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=categories,
            yticklabels=categories
        )
        plt.title(
            f"Hindi News Classification Confusion Matrix\n"
            f"Accuracy: {acc*100:.1f}% | Macro F1: {f1_macro:.2f} | Weighted F1: {f1_wt:.2f}",
            fontsize=12,
            fontweight="bold"
        )
        plt.xlabel("Predicted Category", fontsize=11, fontweight="bold")
        plt.ylabel("Ground Truth Category", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"[Evaluation] Saved confusion matrix plot to: {plot_file}")
        
    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_wt),
        "recall_weighted": float(r_wt),
        "f1_weighted": float(f1_wt),
        "model_loaded": classifier.model_name_or_path,
        "model_type": checkpoint_type,
        "confusion_matrix": cm.tolist(),
        "report": report_str,
        "plot_path": str(plot_file) if plot_file else None
    }


# ==============================================================================
# Standalone Runner
# ==============================================================================

if __name__ == "__main__":
    print("Testing Classification Inference & Evaluation...")
    sample_text = "संसद के विशेष सत्र में विपक्षी दलों ने नई नीतियों और बजट प्रस्तावों पर तीखी बहस की।"
    result = predict_category(sample_text)
    print("\nSample Prediction:")
    print(f"Text:        {sample_text}")
    print(f"Category:    {result['category']} ({result['category_hi']})")
    print(f"Confidence:  {result['confidence']:.2%}")
    print(f"Model:       {result['model_used']}")
    
    print("\nRunning Evaluation on Test Set...")
    eval_res = evaluate_classifier_performance()
