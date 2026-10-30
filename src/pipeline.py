import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG
from src.preprocess import clean_hindi_text, detect_language
from src.classify import get_classifier, HindiNewsClassifier
from src.summarize import get_summarizer, HindiNewsSummarizer
from src.translate import get_translator, MultilingualTranslator


class HindiNewsPipeline:
    """
    End-to-End Multilingual News Processor (Hindi, English, Marathi).
    """
    
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        summarizer_path: Optional[str] = None
    ):
        self.classifier = get_classifier(classifier_path)
        self.summarizer = get_summarizer(summarizer_path)
        self.translator = get_translator()
        
    def preload_all(self):
        """Preload all model weights into memory/GPU for fast warm inference."""
        print("[Pipeline] Preloading models into memory...")
        self.classifier.load_model()
        self.summarizer.load_model()
        self.translator.load_model()
        print("[Pipeline] All models initialized and ready.")

    def run(self, article_text: str, override_lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the end-to-end processing pipeline on a news article (Hindi, English, or Marathi).
        
        Args:
            article_text (str): Input news article in Hindi, English, or Marathi.
            override_lang (str, optional): Manually specified language code ('hi', 'en', 'mr').
            
        Returns:
            dict containing category, language detection info, multi-lingual summaries, and runtime stats.
        """
        start_total = time.perf_counter()
        timing = {}
        
        # 1. Validation & Language Detection
        t0 = time.perf_counter()
        if not isinstance(article_text, str) or not article_text.strip():
            return {
                "status": "error",
                "error_message": "Please enter a valid news article text (Input text is empty).",
                "category": "Unknown",
                "category_hi": "अज्ञात",
                "confidence": 0.0,
                "probabilities": {},
                "detected_language": {"code": "hi", "display": "Hindi (हिंदी)", "flag": "🇮🇳 Hindi"},
                "hindi_summary": "",
                "english_summary": "",
                "marathi_summary": "",
                "execution_times": {"total": 0.0}
            }
            
        lang_info = detect_language(article_text)
        if override_lang in ["hi", "en", "mr"]:
            lang_info["code"] = override_lang
            lang_info["display"] = {"hi": "Hindi (हिंदी)", "en": "English", "mr": "Marathi (मराठी)"}[override_lang]
            lang_info["flag"] = {"hi": "🇮🇳 Hindi", "en": "🇬🇧 English", "mr": "🚩 Marathi"}[override_lang]
            
        lang_code = lang_info["code"]
        cleaned_text = clean_hindi_text(article_text) if lang_code in ["hi", "mr"] else article_text.strip()
        
        if len(cleaned_text) < 15:
            return {
                "status": "error",
                "error_message": "Article text is too short. Please enter at least 15-20 characters.",
                "category": "Unknown",
                "category_hi": "अज्ञात",
                "confidence": 0.0,
                "probabilities": {},
                "detected_language": lang_info,
                "hindi_summary": cleaned_text,
                "english_summary": cleaned_text,
                "marathi_summary": cleaned_text,
                "execution_times": {"total": 0.0}
            }
        timing["preprocess"] = round(time.perf_counter() - t0, 4)
        
        # 2. Topic Classification (Cross-lingual XLM-RoBERTa)
        t0 = time.perf_counter()
        try:
            class_res = self.classifier.predict(cleaned_text)
        except Exception as e:
            print(f"[Pipeline Warning] Classifier failed ({e}), using fallback.")
            class_res = {
                "category": "Politics",
                "category_hi": "राजनीति",
                "confidence": 0.5,
                "probabilities": {c: 0.2 for c in CONFIG.categories},
                "model_used": "Fallback"
            }
        timing["classify"] = round(time.perf_counter() - t0, 4)
        
        # 3. Summarization in Input Language
        t0 = time.perf_counter()
        try:
            sum_res = self.summarizer.summarize(cleaned_text, lang=lang_code)
            input_lang_summary = sum_res["summary"]
            sum_model = sum_res.get("model_used", "Summarizer")
        except Exception as e:
            print(f"[Pipeline Warning] Summarizer failed ({e}), using fallback.")
            term = "." if lang_code == "en" else "।"
            input_lang_summary = cleaned_text[:180] + term
            sum_model = "Fallback"
        timing["summarize"] = round(time.perf_counter() - t0, 4)
        
        # 4. Multi-Directional Translation (NLLB-200 to all remaining languages)
        t0 = time.perf_counter()
        try:
            all_summaries = self.translator.translate_summary(input_lang_summary, src_lang=lang_code)
            hindi_summary = all_summaries.get("hindi", input_lang_summary)
            english_summary = all_summaries.get("english", input_lang_summary)
            marathi_summary = all_summaries.get("marathi", input_lang_summary)
        except Exception as e:
            print(f"[Pipeline Warning] Translator failed ({e}).")
            hindi_summary = input_lang_summary if lang_code == "hi" else "अनुवाद उपलब्ध नहीं है।"
            english_summary = input_lang_summary if lang_code == "en" else "Translation unavailable."
            marathi_summary = input_lang_summary if lang_code == "mr" else "भाषांतर उपलब्ध नाही."
        timing["translate"] = round(time.perf_counter() - t0, 4)
        
        timing["total"] = round(time.perf_counter() - start_total, 4)
        
        primary_sum_len = len(input_lang_summary.split())
        return {
            "status": "success",
            "error_message": None,
            "input_text": article_text,
            "clean_input_text": cleaned_text,
            "detected_language": lang_info,
            "category": class_res["category"],
            "category_hi": class_res["category_hi"],
            "confidence": class_res["confidence"],
            "probabilities": class_res.get("probabilities", {}),
            "hindi_summary": hindi_summary,
            "english_summary": english_summary,
            "marathi_summary": marathi_summary,
            "models_used": {
                "classifier": class_res.get("model_used", "XLM-RoBERTa"),
                "summarizer": sum_model,
                "translator": "Meta NLLB-200 Distilled (Unified Multi-NMT)"
            },
            "execution_times": timing,
            "stats": {
                "input_char_count": len(article_text),
                "input_word_count": len(article_text.split()),
                "original_word_count": len(article_text.split()),
                "summary_char_count": len(input_lang_summary),
                "summary_word_count": primary_sum_len,
                "compression_ratio": f"{(1 - primary_sum_len / max(len(article_text.split()), 1))*100:.1f}%"
            }
        }


# Singleton pipeline
_DEFAULT_PIPELINE: Optional[HindiNewsPipeline] = None

def get_pipeline() -> HindiNewsPipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = HindiNewsPipeline()
    return _DEFAULT_PIPELINE


def run_pipeline(article_text: str) -> Dict[str, Any]:
    """
    Main entry point for end-to-end processing.
    """
    pipeline = get_pipeline()
    return pipeline.run(article_text)


# ==============================================================================
# Standalone Test
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("TESTING END-TO-END HINDI NEWS PIPELINE")
    print("=" * 65)
    
    sample_news = (
        "संसद के बजट सत्र के दूसरे चरण की शुरुआत हंगामेदार रही। विपक्षी दलों ने महंगाई, "
        "बेरोजगारी और हालिया नीतिगत फैसलों को लेकर सरकार के खिलाफ नारेबाजी की। दोनों सदनों लोकसभा और "
        "राज्यसभा में कार्यवाही बार-बार स्थगित करनी पड़ी। सरकार ने कहा कि वह सभी जरूरी मुद्दों पर "
        "नियमानुसार चर्चा के लिए तैयार है लेकिन विपक्ष जानबूझकर संसद की कार्यवाही बाधित कर रहा है। "
        "लोकसभा अध्यक्ष ने सभी सांसदों से सदन की गरिमा बनाए रखने और जनहित के मुद्दों पर सार्थक बहस करने की अपील की।"
    )
    
    output = run_pipeline(sample_news)
    
    print("\n[RESULT]")
    print(f"Status:             {output['status']}")
    print(f"Predicted Category: {output['category']} ({output['category_hi']}) [Confidence: {output['confidence']:.2%}]")
    print(f"\nHindi Summary:      {output['hindi_summary']}")
    print(f"English Summary:    {output['english_summary']}")
    print(f"Marathi Summary:    {output['marathi_summary']}")
    print(f"\nStats:              {output['stats']}")
    print(f"Latencies:          {output['execution_times']}")
    print("=" * 65)
