import re
import os
import sys
import unicodedata
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import langdetect
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split

DetectorFactory.seed = 0

# Language mapping dictionaries
LANG_DISPLAY_MAP = {
    "hi": "Hindi (हिंदी)",
    "en": "English",
    "mr": "Marathi (मराठी)"
}

LANG_FLAG_MAP = {
    "hi": "🇮🇳 Hindi",
    "en": "🇬🇧 English",
    "mr": "🚩 Marathi"
}

# Distinguishing vocabulary markers
HI_STOPWORDS = {"है", "हैं", "था", "थी", "थे", "कि", "ने", "का", "के", "की", "में", "और", "पर", "से", "को", "यह", "वह", "द्वारा"}
MR_STOPWORDS = {"आहे", "आहीत", "होता", "होती", "होते", "की", "त्यांनी", "च्या", "मध्ये", "आणि", "वर", "वरून", "झाला", "झाली", "केले", "मिळाला", "पर्यंत"}

def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect whether the input text is in Hindi (hi), English (en), or Marathi (mr).
    Uses a hybrid approach combining script character analysis, stopword frequency, and langdetect.
    """
    if not isinstance(text, str) or not text.strip():
        return {"code": "hi", "display": "Hindi (हिंदी)", "flag": "🇮🇳 Hindi", "confidence": 1.0, "is_reliable": True}
        
    cleaned = text.strip()
    
    # 1. Script-based check: Latin script vs Devanagari script
    latin_chars = len(re.findall(r"[a-zA-Z]", cleaned))
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", cleaned))
    total_alpha = latin_chars + devanagari_chars + 1e-5
    
    if latin_chars / total_alpha > 0.6:
        return {"code": "en", "display": "English", "flag": "🇬🇧 English", "confidence": 0.98, "is_reliable": True}
        
    # 2. Devanagari Vocabulary Stopword Ratio Disambiguation (Hindi vs Marathi)
    words = set(re.findall(r"[\u0900-\u097F]+", cleaned))
    hi_count = len(words.intersection(HI_STOPWORDS))
    mr_count = len(words.intersection(MR_STOPWORDS))
    
    if mr_count > hi_count:
        return {"code": "mr", "display": "Marathi (मराठी)", "flag": "🚩 Marathi", "confidence": 0.95, "is_reliable": True}
    elif hi_count > mr_count:
        return {"code": "hi", "display": "Hindi (हिंदी)", "flag": "🇮🇳 Hindi", "confidence": 0.95, "is_reliable": True}
        
    # 3. Fallback to langdetect library
    try:
        raw_pred = detect(cleaned)
        if raw_pred in ["mr", "mar"]:
            code = "mr"
        elif raw_pred in ["en", "eng"]:
            code = "en"
        else:
            code = "hi"
    except Exception:
        code = "hi"
        
    return {
        "code": code,
        "display": LANG_DISPLAY_MAP.get(code, "Hindi (हिंदी)"),
        "flag": LANG_FLAG_MAP.get(code, "🇮🇳 Hindi"),
        "confidence": 0.90,
        "is_reliable": True
    }

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG, PROJECT_ROOT


# ==============================================================================
# Devanagari Unicode Normalization Utilities
# ==============================================================================

# Nukta mapping for standardized Devanagari representation
NUKTA_MAP = {
    "\u0915\u093c": "\u0958",  # क़
    "\u0916\u093c": "\u0959",  # ख़
    "\u0917\u093c": "\u095a",  # ग़
    "\u091c\u093c": "\u095b",  # ज़
    "\u0921\u093c": "\u095c",  # ड़
    "\u0922\u093c": "\u095d",  # ढ़
    "\u092b\u093c": "\u095e",  # फ़
    "\u092f\u093c": "\u095f",  # य़
}

# Regex patterns for text cleaning
HTML_TAG_REGEX = re.compile(r"<[^>]+>")
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
WHITESPACE_REGEX = re.compile(r"\s+")
SPECIAL_CHAR_REGEX = re.compile(r"[^\w\s\d\u0900-\u097F।॥\.,?!'-]")


def normalize_devanagari(text: str) -> str:
    """
    Standardize Devanagari Unicode characters.
    Applies NFC normalization, Nukta unification, and removes invisible format characters.
    """
    if not isinstance(text, str):
        return ""
    
    # Standard Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Replace Nukta character sequences with unified single codepoints
    for non_nukta, nukta in NUKTA_MAP.items():
        text = text.replace(non_nukta, nukta)
        
    # Remove Zero Width Joiner (ZWJ) and Zero Width Non-Joiner (ZWNJ) if isolated noise
    # (preserve if part of standard devanagari ligature contexts)
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\ufeff", "")  # BOM
    
    return text


def clean_hindi_text(text: str, preserve_punctuation: bool = True) -> str:
    """
    Clean Hindi news text by removing HTML tags, URLs, emails, and irrelevant symbols.
    
    Args:
        text (str): Input raw Hindi text.
        preserve_punctuation (bool): Whether to keep sentence boundaries (danda ।, ?, !).
        
    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Unicode normalization
    text = normalize_devanagari(text)
    
    # 2. Remove HTML tags
    text = HTML_TAG_REGEX.sub(" ", text)
    
    # 3. Remove URLs and Emails
    text = URL_REGEX.sub(" ", text)
    text = EMAIL_REGEX.sub(" ", text)
    
    # 4. Remove unprintable and exotic symbols while preserving Devanagari range
    if preserve_punctuation:
        text = SPECIAL_CHAR_REGEX.sub(" ", text)
    else:
        text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
        
    # 5. Normalize whitespace
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    
    return text


# ==============================================================================
# Dataset Loaders & Splitters
# ==============================================================================

def load_and_clean_classification_data(
    filepath: Optional[str] = None
) -> pd.DataFrame:
    """
    Load raw classification dataset, perform data validation, deduplication,
    cleaning, and map labels to standard categories.
    Prioritizes full datasets (e.g. Inshorts Hindi ~27k rows) over sample stubs.
    """
    path = None
    if filepath:
        p = Path(filepath)
        if p.exists():
            path = p
        elif (PROJECT_DIR / filepath).exists():
            path = PROJECT_DIR / filepath
        elif (Path.cwd() / filepath).exists():
            path = Path.cwd() / filepath
            
    if path is None:
        # Search for full real datasets first
        raw_candidates = [
            PROJECT_DIR / "data" / "raw" / "inshorts_hindi_real.csv",
            PROJECT_DIR / "data" / "raw" / "hindi_news_inshorts.csv",
            PROJECT_DIR / "data" / "raw" / "inshorts_dataset.csv",
            PROJECT_DIR / "data" / "raw" / "hindi_news.csv",
            CONFIG.raw_data_dir / "inshorts_hindi_real.csv",
            CONFIG.raw_data_dir / "hindi_news.csv",
            Path("/content/PROJ NLP/data/raw/inshorts_hindi_real.csv"),
            Path("/content/data/raw/inshorts_hindi_real.csv"),
            Path.cwd() / "data" / "raw" / "inshorts_hindi_real.csv",
        ]
        for cand in raw_candidates:
            if cand.exists():
                path = cand
                break
                
        # If not found in known names, look for any large CSV in raw_data_dir
        if path is None and CONFIG.raw_data_dir.exists():
            for f in CONFIG.raw_data_dir.glob("*.csv"):
                if "sample" not in f.name.lower() and f.stat().st_size > 100_000:
                    path = f
                    break

        # Fallback to sample stub if nothing else exists
        if path is None:
            path = CONFIG.raw_data_dir / "sample_hindi_news.csv"
            if not path.exists() and (PROJECT_DIR / "data" / "raw" / "sample_hindi_news.csv").exists():
                path = PROJECT_DIR / "data" / "raw" / "sample_hindi_news.csv"
    
    if path is None or not path.exists():
        raise FileNotFoundError(f"Classification dataset not found. Checked candidate locations in '{CONFIG.raw_data_dir}'")
        
    print(f"[Preprocessing] Loading classification dataset from: {path}")
    df = pd.read_csv(path)
    
    # Determine text and label columns
    text_col = None
    for c in ["article", "Content", "content", "text", "body", "news", "clean_text"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
        
    label_col = None
    for c in ["category", "Category", "label", "topic", "tag", "Category_hi"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    # Drop rows with nulls
    df = df.dropna(subset=[text_col, label_col]).copy()
    
    # Clean text column
    df["clean_text"] = df[text_col].apply(clean_hindi_text)
    
    # Filter out empty or extremely short texts (less than 10 characters)
    df = df[df["clean_text"].str.len() > 10].copy()
    
    # Deduplicate
    df = df.drop_duplicates(subset=["clean_text"]).copy()
    
    # Map category names to standard format if needed
    category_alias_map = {
        "politics": "Politics", "राजनीति": "Politics",
        "sports": "Sports", "खेल": "Sports",
        "business": "Business", "व्यापार": "Business", "कारोबार": "Business",
        "entertainment": "Entertainment", "मनोरंजन": "Entertainment",
        "technology": "Technology", "tech": "Technology", "प्रौद्योगिकी": "Technology", "तकनीक": "Technology"
    }
    
    df["category"] = df[label_col].astype(str).map(
        lambda x: category_alias_map.get(x.strip().lower(), x.strip())
    )
    
    # Assign numerical label IDs based on CONFIG
    df["label_id"] = df["category"].map(CONFIG.label2id)
    
    # Drop any categories not in config
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    
    return df[["clean_text", "category", "label_id"]]


def prepare_classification_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into stratified Train, Validation, and Test sets.
    """
    # Check if dataset has sufficient samples per class for stratification
    min_class_count = df["label_id"].value_counts().min()
    stratify = df["label_id"] if min_class_count >= 2 else None
    
    if len(df) <= 15:
        # Small sample dataset: split proportionally without dropping small classes
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )
        val_df = test_df.copy()
        return train_df, val_df, test_df
        
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    val_ratio_adjusted = val_size / (1.0 - test_size)
    stratify_val = train_val_df["label_id"] if (train_val_df["label_id"].value_counts().min() >= 2) else None
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_adjusted, random_state=random_state, stratify=stratify_val
    )
    
    return train_df, val_df, test_df


def save_processed_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Save processed splits to CSV files.
    """
    out_path = output_dir if output_dir else CONFIG.processed_data_dir
    out_path.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "train": out_path / "train_classify.csv",
        "val": out_path / "val_classify.csv",
        "test": out_path / "test_classify.csv",
    }
    
    train_df.to_csv(paths["train"], index=False, encoding="utf-8")
    val_df.to_csv(paths["val"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8")
    
    print(f"[Preprocessing] Successfully saved processed splits:")
    print(f"  - Train ({len(train_df)} rows): {paths['train']}")
    print(f"  - Val   ({len(val_df)} rows): {paths['val']}")
    print(f"  - Test  ({len(test_df)} rows): {paths['test']}")
    
    return paths


def load_and_clean_summarization_data(
    filepath: Optional[str] = None
) -> pd.DataFrame:
    """
    Load summarization and translation dataset (XL-Sum Hindi subset format).
    """
    path = Path(filepath) if filepath else CONFIG.raw_data_dir / "sample_xlsum_hindi.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Summarization dataset not found at {path}")
        
    df = pd.read_csv(path)
    
    # Clean text and summaries
    if "text" in df.columns:
        df["clean_text"] = df["text"].apply(clean_hindi_text)
    if "summary_hi" in df.columns:
        df["clean_summary_hi"] = df["summary_hi"].apply(clean_hindi_text)
        
    return df


# ==============================================================================
# Main execution for testing & preprocessing
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hindi NLP Pipeline: Preprocessing & Data Splits")
    print("=" * 60)
    
    # 1. Test Devanagari text normalization
    sample_raw = " <p>संसद में <b>विपक्ष</b> का जोरदार हंगामा! https://news.com/article?id=101 </p>  "
    sample_cleaned = clean_hindi_text(sample_raw)
    print(f"Raw Input:     '{sample_raw}'")
    print(f"Cleaned Text:  '{sample_cleaned}'")
    print("-" * 60)
    
    # 2. Load and process classification data
    print("[1/2] Processing Hindi Classification Dataset...")
    df_classify = load_and_clean_classification_data()
    print(f"Loaded {len(df_classify)} valid classified articles across classes:")
    print(df_classify["category"].value_counts().to_string())
    
    # 3. Create splits
    train_df, val_df, test_df = prepare_classification_splits(df_classify)
    save_processed_datasets(train_df, val_df, test_df)
    
    # 4. Load summarization dataset
    print("\n[2/2] Processing Hindi Summarization / Translation Dataset...")
    df_sum = load_and_clean_summarization_data()
    print(f"Loaded {len(df_sum)} summarization benchmark samples.")
    print("=" * 60)
    print("Preprocessing completed successfully.")
