import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import yaml

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def get_device():
    """
    Automatically detect the best available device for compute.
    Returns:
        torch.device or str: 'cuda', 'mps' (Apple Silicon), or 'cpu'.
    """
    if HAS_TORCH and torch is not None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        try:
            # Optimize CPU threads on Windows to prevent thread contention
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass
        return torch.device("cpu")
    return "cpu"


@dataclass
class ClassificationConfig:
    model_name: str = "xlm-roberta-base"
    fallback_model: str = "ai4bharat/indic-bert"
    saved_model_path: str = "models/classifier/best_model"
    max_seq_length: int = 256
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = False


@dataclass
class SummarizationConfig:
    model_name: str = "ai4bharat/IndicBART"
    fallback_model: str = "csebuetnlp/mT5_multilingual_XLSum"
    lightweight_model: str = "google/mt5-small"
    saved_model_path: str = "models/summarizer/best_model"
    max_input_length: int = 384
    max_target_length: int = 70
    min_target_length: int = 20
    num_beams: int = 2
    length_penalty: float = 0.8
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True


@dataclass
class TranslationConfig:
    indic_en_model: str = "ai4bharat/indictrans2-indic-en-1B"
    indic_indic_model: str = "ai4bharat/indictrans2-indic-indic-1B"
    fast_en_model: str = "Helsinki-NLP/opus-mt-hi-en"
    fast_mr_model: str = "facebook/mbart-large-50-many-to-many-mmt"
    saved_model_path: str = "models/translator/"
    max_length: int = 90
    num_beams: int = 1


@dataclass
class PipelineConfig:
    project_name: str = "Multilingual Hindi News NLP Pipeline"
    version: str = "1.0.0"
    seed: int = 42
    device: str = "auto"
    
    categories: List[str] = field(default_factory=lambda: [
        "Politics", "Sports", "Business", "Entertainment", "Technology"
    ])
    categories_hi: List[str] = field(default_factory=lambda: [
        "राजनीति", "खेल", "व्यापार", "मनोरंजन", "प्रौद्योगिकी"
    ])
    label2id: Dict[str, int] = field(default_factory=lambda: {
        "Politics": 0, "Sports": 1, "Business": 2, "Entertainment": 3, "Technology": 4
    })
    id2label: Dict[int, str] = field(default_factory=lambda: {
        0: "Politics", 1: "Sports", 2: "Business", 3: "Entertainment", 4: "Technology"
    })
    
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from YAML file and return typed PipelineConfig instance.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if not path.exists():
        return PipelineConfig()
    
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    cfg = PipelineConfig()
    
    # Project info
    if "project" in raw:
        cfg.project_name = raw["project"].get("name", cfg.project_name)
        cfg.version = raw["project"].get("version", cfg.version)
        cfg.seed = raw["project"].get("seed", cfg.seed)
        cfg.device = raw["project"].get("device", cfg.device)
        
    # Categories
    if "categories" in raw:
        cats = raw["categories"]
        cfg.categories = cats.get("labels", cfg.categories)
        cfg.categories_hi = cats.get("labels_hi", cfg.categories_hi)
        cfg.label2id = cats.get("label2id", cfg.label2id)
        # Ensure int keys for id2label
        if "id2label" in cats:
            cfg.id2label = {int(k): v for k, v in cats["id2label"].items()}
            
    # Classification
    if "classification" in raw:
        c = raw["classification"]
        cfg.classification = ClassificationConfig(
            model_name=c.get("model_name", cfg.classification.model_name),
            fallback_model=c.get("fallback_model", cfg.classification.fallback_model),
            saved_model_path=c.get("saved_model_path", cfg.classification.saved_model_path),
            max_seq_length=c.get("max_seq_length", cfg.classification.max_seq_length),
            batch_size=c.get("batch_size", cfg.classification.batch_size),
            eval_batch_size=c.get("eval_batch_size", cfg.classification.eval_batch_size),
            learning_rate=float(c.get("learning_rate", cfg.classification.learning_rate)),
            num_epochs=int(c.get("num_epochs", cfg.classification.num_epochs)),
            weight_decay=float(c.get("weight_decay", cfg.classification.weight_decay)),
            warmup_ratio=float(c.get("warmup_ratio", cfg.classification.warmup_ratio)),
            fp16=bool(c.get("fp16", cfg.classification.fp16)),
        )
        
    # Summarization
    if "summarization" in raw:
        s = raw["summarization"]
        cfg.summarization = SummarizationConfig(
            model_name=s.get("model_name", cfg.summarization.model_name),
            fallback_model=s.get("fallback_model", cfg.summarization.fallback_model),
            lightweight_model=s.get("lightweight_model", cfg.summarization.lightweight_model),
            saved_model_path=s.get("saved_model_path", cfg.summarization.saved_model_path),
            max_input_length=s.get("max_input_length", cfg.summarization.max_input_length),
            max_target_length=s.get("max_target_length", cfg.summarization.max_target_length),
            min_target_length=s.get("min_target_length", cfg.summarization.min_target_length),
            num_beams=s.get("num_beams", cfg.summarization.num_beams),
            length_penalty=float(s.get("length_penalty", cfg.summarization.length_penalty)),
            no_repeat_ngram_size=s.get("no_repeat_ngram_size", cfg.summarization.no_repeat_ngram_size),
            early_stopping=bool(s.get("early_stopping", cfg.summarization.early_stopping)),
        )
        
    # Translation
    if "translation" in raw:
        t = raw["translation"]
        cfg.translation = TranslationConfig(
            indic_en_model=t.get("indic_en_model", cfg.translation.indic_en_model),
            indic_indic_model=t.get("indic_indic_model", cfg.translation.indic_indic_model),
            fast_en_model=t.get("fast_en_model", cfg.translation.fast_en_model),
            fast_mr_model=t.get("fast_mr_model", cfg.translation.fast_mr_model),
            saved_model_path=t.get("saved_model_path", cfg.translation.saved_model_path),
            max_length=t.get("max_length", cfg.translation.max_length),
            num_beams=t.get("num_beams", cfg.translation.num_beams),
        )
        
    # Ensure directories exist
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    return cfg


# Global singleton config
CONFIG = load_config()
DEVICE = get_device() if CONFIG.device == "auto" else (torch.device(CONFIG.device) if HAS_TORCH and torch else CONFIG.device)
