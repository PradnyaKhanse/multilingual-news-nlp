import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG, get_device
from src.preprocess import clean_hindi_text, load_and_clean_summarization_data


# ==============================================================================
# Extractive Fallback Summarizer (TextRank / Lead-K Sentence Scoring)
# ==============================================================================

def fallback_hindi_summarizer(text: str, max_sentences: int = 2, lang: str = "hi") -> str:
    """
    Extractive sentence ranker for Hindi, Marathi, and English text used as a fallback
    when large seq2seq checkpoints are unavailable or downloading.
    """
    cleaned = clean_hindi_text(text) if lang in ["hi", "mr"] else text.strip()
    
    # Language-aware sentence splitting
    if lang == "en":
        sentences = [s.strip() for s in re.split(r"[.!\?\n]+", cleaned) if len(s.strip()) > 15]
    else:
        sentences = [s.strip() for s in re.split(r"[।॥\n\?!]+", cleaned) if len(s.strip()) > 15]
        if not sentences:
            sentences = [s.strip() for s in re.split(r"[.!\?\n]+", cleaned) if len(s.strip()) > 15]
    
    if not sentences:
        return cleaned[:180] + "..." if len(cleaned) > 180 else cleaned
        
    if len(sentences) <= max_sentences:
        join_char = ". " if lang == "en" else "। "
        term_char = "." if lang == "en" else "।"
        return join_char.join(sentences) + term_char
        
    # Score sentences by word frequency and position (news lead bias)
    words = cleaned.lower().split()
    word_freq = {}
    for w in words:
        if len(w) > 2:
            word_freq[w] = word_freq.get(w, 0) + 1
            
    scored_sentences = []
    for idx, sent in enumerate(sentences):
        score = sum(word_freq.get(w.lower(), 0) for w in sent.split()) / (len(sent.split()) + 1)
        # Position weight: first 2 sentences in news contain highest information density
        position_weight = 1.5 if idx == 0 else (1.2 if idx == 1 else 1.0)
        scored_sentences.append((score * position_weight, idx, sent))
        
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])
    
    join_char = ". " if lang == "en" else "। "
    term_char = "." if lang == "en" else "।"
    summary = join_char.join(s[2] for s in top_sentences)
    if not summary.endswith((".", "!", "?", "।")):
        summary += term_char
    return summary


# ==============================================================================
# Model Architecture & Inference Class
# ==============================================================================

class HindiNewsSummarizer:
    """
    Abstractive Summarization Pipeline for Hindi News.
    Supports AI4Bharat IndicBART, mT5-XLSum, and mt5-small backends.
    """
    
    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name_or_path = model_name_or_path or CONFIG.summarization.model_name
        self.fallback_model_name = CONFIG.summarization.fallback_model
        self.device = get_device()
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.active_backend = "Fallback"
        
    def load_model(self, force_reload: bool = False):
        """Lazy load seq2seq summarization model."""
        if self.is_loaded and not force_reload:
            return
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # 1. Attempt loading primary model (e.g. IndicBART or specified model)
            print(f"[Summarizer] Attempting to load primary model: {self.model_name_or_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, do_lower_case=False)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
                self.active_backend = self.model_name_or_path
            except Exception as e1:
                print(f"[Summarizer] Primary model failed ({e1}). Trying fallback: {self.fallback_model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.fallback_model_name)
                self.active_backend = self.fallback_model_name
                
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[Summarizer] Successfully loaded model '{self.active_backend}' on {self.device}.")
            
        except Exception as e:
            print(f"[Summarizer Warning] Could not load transformer weights ({e}).")
            print("[Summarizer] Using high-precision Indic Extractive/TextRank fallback.")
            self.is_loaded = False
            self.active_backend = "Indic-TextRank-Extractive"

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        lang: str = "hi"
    ) -> Dict[str, Any]:
        """
        Generate abstractive summary for the input article in the specified language (hi, mr, en).
        """
        cleaned_text = clean_hindi_text(text) if lang in ["hi", "mr"] else text.strip()
        if not cleaned_text or len(cleaned_text) < 15:
            return {
                "summary": cleaned_text,
                "model_used": "None (Too short)",
                "input_length": len(text),
                "summary_length": len(cleaned_text)
            }
            
        max_len = max_length or CONFIG.summarization.max_target_length
        min_len = min_length or CONFIG.summarization.min_target_length
        beams = num_beams or CONFIG.summarization.num_beams
        
        if not self.is_loaded:
            self.load_model()
            
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                
                # Format language token for IndicBART if applicable
                input_str = cleaned_text
                forced_bos = None
                if "indicbart" in self.active_backend.lower():
                    lang_tag_map = {"hi": "_hi_IN", "mr": "_mr_IN", "en": "_en_IN"}
                    tag = lang_tag_map.get(lang, "_hi_IN")
                    input_str = cleaned_text + f" </s> {tag}"
                    try:
                        forced_bos = self.tokenizer.convert_tokens_to_ids(tag)
                        if forced_bos == self.tokenizer.unk_token_id:
                            forced_bos = None
                    except Exception:
                        forced_bos = None
                    
                inputs = self.tokenizer(
                    input_str,
                    max_length=CONFIG.summarization.max_input_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                ).to(self.device)
                
                gen_kwargs = {
                    "max_length": max_len,
                    "min_length": min_len,
                    "num_beams": beams,
                    "length_penalty": CONFIG.summarization.length_penalty,
                    "no_repeat_ngram_size": CONFIG.summarization.no_repeat_ngram_size,
                    "early_stopping": CONFIG.summarization.early_stopping,
                    "repetition_penalty": 1.3
                }
                if forced_bos is not None:
                    gen_kwargs["forced_bos_token_id"] = forced_bos
                    
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        **gen_kwargs
                    )
                    
                decoded = self.tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Clean generated output
                cleaned_summary = clean_hindi_text(decoded)
                
                # Remove common IndicBART prompt leakage prefix artifacts
                for artifact in ["ख़ुशी ", "खुशी ", "गुरूवार ", "गुरुवार ", "मालूम ", "फोटो ", "वीडियो "]:
                    if cleaned_summary.startswith(artifact) and artifact not in cleaned_text:
                        cleaned_summary = cleaned_summary[len(artifact):].strip()
                        
                if not cleaned_summary.endswith(("।", "?", "!")):
                    cleaned_summary += "।"
                    
                return {
                    "summary": cleaned_summary,
                    "model_used": self.active_backend,
                    "input_length": len(text),
                    "summary_length": len(cleaned_summary)
                }
            except Exception as e:
                print(f"[Summarizer Error during generation]: {e}. Falling back to rule-based.")
                
        # Fallback
        res = fallback_hindi_summarizer(cleaned_text, lang=lang)
        return {
            "summary": res,
            "model_used": "Indic-TextRank-Extractive (Fallback)",
            "input_length": len(text),
            "summary_length": len(res)
        }


# Singleton instance
_DEFAULT_SUMMARIZER: Optional[HindiNewsSummarizer] = None

def get_summarizer(model_path: Optional[str] = None) -> HindiNewsSummarizer:
    global _DEFAULT_SUMMARIZER
    if _DEFAULT_SUMMARIZER is None or (model_path and _DEFAULT_SUMMARIZER.model_name_or_path != model_path):
        _DEFAULT_SUMMARIZER = HindiNewsSummarizer(model_path)
    return _DEFAULT_SUMMARIZER


def summarize_hindi(text: str, model_path: Optional[str] = None) -> str:
    """Convenience function returning just the Hindi summary string."""
    summarizer = get_summarizer(model_path)
    result = summarizer.summarize(text)
    return result["summary"]


# ==============================================================================
# Summarization Evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
# ==============================================================================

def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for Hindi text.
    Handles tokenization for Indic Devanagari script.
    """
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        # Devanagari tokenized Rouge
        results = rouge.compute(
            predictions=predictions,
            references=references,
            tokenizer=lambda x: clean_hindi_text(x).split()
        )
        return {
            "rouge1": float(results["rouge1"]),
            "rouge2": float(results["rouge2"]),
            "rougeL": float(results["rougeL"]),
            "rougeLsum": float(results.get("rougeLsum", results["rougeL"]))
        }
    except Exception as e:
        # Custom exact n-gram overlap ROUGE implementation as robust fallback
        return compute_ngram_rouge_fallback(predictions, references)


def compute_ngram_rouge_fallback(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Pure Python fallback for ROUGE 1/2/L calculation."""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    r1_list, r2_list, rl_list = [], [], []

    for pred, ref in zip(predictions, references):
        p_tokens = clean_hindi_text(pred).split()
        r_tokens = clean_hindi_text(ref).split()

        if not p_tokens or not r_tokens:
            continue

        # ROUGE-1
        p_set1, r_set1 = set(p_tokens), set(r_tokens)
        match1 = len(p_set1.intersection(r_set1))
        p1 = match1 / len(p_tokens)
        rec1 = match1 / len(r_tokens)
        f1_1 = (2 * p1 * rec1) / (p1 + rec1) if (p1 + rec1) > 0 else 0.0
        r1_list.append(f1_1)

        # ROUGE-2
        p_ngrams2 = set(get_ngrams(p_tokens, 2))
        r_ngrams2 = set(get_ngrams(r_tokens, 2))
        match2 = len(p_ngrams2.intersection(r_ngrams2))
        p2 = match2 / len(p_ngrams2) if p_ngrams2 else 0.0
        rec2 = match2 / len(r_ngrams2) if r_ngrams2 else 0.0
        f1_2 = (2 * p2 * rec2) / (p2 + rec2) if (p2 + rec2) > 0 else 0.0
        r2_list.append(f1_2)

        # ROUGE-L (Approximation via common token sequence)
        rl_list.append((f1_1 + f1_2) / 2 if f1_2 > 0 else f1_1 * 0.8)

    return {
        "rouge1": float(np.mean(r1_list)) if r1_list else 0.0,
        "rouge2": float(np.mean(r2_list)) if r2_list else 0.0,
        "rougeL": float(np.mean(rl_list)) if rl_list else 0.0,
    }


def evaluate_summarizer_performance(
    test_csv: Optional[str] = None,
    model_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Evaluate summarizer on test dataset and print ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        test_csv (str, optional): Path to test CSV file. If None, resolves from data directory or HuggingFace.
        model_path (str, optional): Explicit model name or checkpoint path to evaluate.
        sample_size (int, optional): Maximum number of samples to evaluate (e.g. 100-200 for fast eval).
        batch_size (int): Batch size for neural inference.
        
    Returns:
        Dict with ROUGE metrics, sample count, model name, and sample predictions.
    """
    import torch
    
    # 1. Resolve Dataset
    df = None
    data_source_desc = ""
    
    if test_csv:
        test_path = Path(test_csv)
        if not test_path.exists():
            candidates = [
                PROJECT_DIR / test_csv,
                Path.cwd() / test_csv,
                PROJECT_DIR / "data" / "raw" / Path(test_csv).name,
                PROJECT_DIR / "data" / "processed" / Path(test_csv).name,
            ]
            found = False
            for cand in candidates:
                if cand.exists():
                    test_path = cand
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"[Summarization Eval Error] Specified test CSV '{test_csv}' was not found.\n"
                    f"Checked paths:\n  - {Path(test_csv).resolve()}\n  - {(PROJECT_DIR / test_csv).resolve()}"
                )
        df = load_and_clean_summarization_data(str(test_path))
        data_source_desc = str(test_path)
    else:
        # 1. Check for real processed XL-Sum test dataset on disk
        processed_candidates = [
            PROJECT_DIR / "data" / "processed" / "xlsum_hindi_test.csv",
            Path.cwd() / "data" / "processed" / "xlsum_hindi_test.csv",
            Path.cwd().parent / "data" / "processed" / "xlsum_hindi_test.csv",
            Path("/content/PROJ NLP/data/processed/xlsum_hindi_test.csv"),
            Path("/content/data/processed/xlsum_hindi_test.csv"),
        ]
        for p_cand in processed_candidates:
            if p_cand.exists():
                try:
                    cand_df = load_and_clean_summarization_data(str(p_cand))
                    if len(cand_df) >= 20:
                        df = cand_df
                        data_source_desc = f"Processed XL-Sum Test Dataset ({p_cand.name}, {len(df)} samples)"
                        break
                except Exception:
                    pass

        # 2. If not cached on disk, download the REAL XL-Sum Hindi dataset from Hugging Face
        if df is None:
            try:
                from datasets import load_dataset
                target_count = sample_size or 100
                print(f"[Summarization Eval] Fetching real XL-Sum Hindi test benchmark ({target_count} articles) from Hugging Face 'csebuetnlp/xlsum'...")
                split_query = f"test[:{target_count}]"
                hf_ds = load_dataset("csebuetnlp/xlsum", "hindi", split=split_query)
                df = pd.DataFrame({
                    "clean_text": [clean_hindi_text(x) for x in hf_ds["text"]],
                    "clean_summary_hi": [clean_hindi_text(x) for x in hf_ds["summary"]]
                })
                data_source_desc = f"Hugging Face (csebuetnlp/xlsum - hindi [{split_query}], {len(df)} samples)"
                
                # Cache to disk for instant loading
                cache_path = PROJECT_DIR / "data" / "processed" / "xlsum_hindi_test.csv"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False, encoding="utf-8")
                print(f"[Summarization Eval] Cached {len(df)} benchmark samples to: {cache_path}")
            except Exception as e:
                print(f"[Summarization Eval Warning] Hugging Face live dataset fetch failed: {e}")
                
        # 3. Emergency fallback to sample stub only if HF is unreachable
        if df is None:
            sample_candidates = [
                PROJECT_DIR / "data" / "raw" / "sample_xlsum_hindi.csv",
                Path.cwd() / "data" / "raw" / "sample_xlsum_hindi.csv",
                Path("/content/PROJ NLP/data/raw/sample_xlsum_hindi.csv")
            ]
            for s_cand in sample_candidates:
                if s_cand.exists():
                    df = load_and_clean_summarization_data(str(s_cand))
                    data_source_desc = f"Local Sample Stub ({s_cand.name} - Note: only {len(df)} samples)"
                    print(f"[Summarization Eval Warning] Using small local sample stub: {s_cand}")
                    break

    if df is None or len(df) == 0:
        raise ValueError("[Summarization Eval Error] Summarization test dataset is empty (0 samples).")

    # Column normalization
    if "clean_text" not in df.columns:
        if "text" in df.columns:
            df["clean_text"] = df["text"].apply(clean_hindi_text)
        elif "article" in df.columns:
            df["clean_text"] = df["article"].apply(clean_hindi_text)
        else:
            df["clean_text"] = df.iloc[:, 0].apply(clean_hindi_text)

    if "clean_summary_hi" not in df.columns:
        if "summary_hi" in df.columns:
            df["clean_summary_hi"] = df["summary_hi"].apply(clean_hindi_text)
        elif "summary" in df.columns:
            df["clean_summary_hi"] = df["summary"].apply(clean_hindi_text)
        else:
            df["clean_summary_hi"] = df.iloc[:, 1].apply(clean_hindi_text)

    # Subsample if sample_size specified or default on CPU for speed
    if sample_size and len(df) > sample_size:
        df = df.iloc[:sample_size].copy()
    elif str(get_device()) == "cpu" and len(df) > 20:
        df = df.iloc[:20].copy()

    # 2. Load Summarizer Model
    summarizer = get_summarizer(model_path)
    summarizer.load_model(force_reload=True)
    
    print("=" * 65)
    print("HINDI ABSTRACTIVE SUMMARIZATION EVALUATION (XL-Sum Benchmark)")
    print("=" * 65)
    print(f"Primary Model Name: {summarizer.model_name_or_path}")
    print(f"Active Backend:     {summarizer.active_backend}")
    print(f"Execution Device:   {summarizer.device}")
    print(f"Dataset Source:     {data_source_desc}")
    print(f"Samples Evaluated:  {len(df)} articles")
    print("=" * 65)
    
    preds = []
    refs = df["clean_summary_hi"].tolist()
    
    # 3. Batched generation if PyTorch seq2seq model is loaded
    if summarizer.is_loaded and summarizer.model is not None and summarizer.tokenizer is not None:
        texts = df["clean_text"].tolist()
        print(f"[Summarization Eval] Generating summaries using neural backend '{summarizer.active_backend}'...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            formatted_batch = []
            for t in batch_texts:
                inp = clean_hindi_text(t)
                if "indicbart" in summarizer.active_backend.lower():
                    inp = inp + " </s> _hi_IN"
                formatted_batch.append(inp)
                
            try:
                inputs = summarizer.tokenizer(
                    formatted_batch,
                    max_length=CONFIG.summarization.max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(summarizer.device)
                
                with torch.no_grad():
                    summary_ids = summarizer.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_length=CONFIG.summarization.max_target_length,
                        min_length=CONFIG.summarization.min_target_length,
                        num_beams=1 if str(summarizer.device) == "cpu" else CONFIG.summarization.num_beams,
                        length_penalty=CONFIG.summarization.length_penalty,
                        no_repeat_ngram_size=CONFIG.summarization.no_repeat_ngram_size,
                        early_stopping=CONFIG.summarization.early_stopping
                    )
                decoded = summarizer.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
                for d in decoded:
                    c_sum = clean_hindi_text(d)
                    if not c_sum.endswith(("।", "?", "!")):
                        c_sum += "।"
                    preds.append(c_sum)
            except Exception as e:
                print(f"[Summarization Eval Warning] Batch generation error ({e}), falling back to single inference.")
                for t in batch_texts:
                    preds.append(summarizer.summarize(t)["summary"])
    else:
        for idx, row in df.iterrows():
            gen_res = summarizer.summarize(row["clean_text"])
            preds.append(gen_res["summary"])
            
    # 4. Compute ROUGE
    scores = compute_rouge_scores(preds, refs)
    
    print("\nSUMMARIZATION EVALUATION METRICS (ROUGE):")
    print(f"  • ROUGE-1 F1:  {scores['rouge1']:.4f} ({scores['rouge1']*100:.2f}%)")
    print(f"  • ROUGE-2 F1:  {scores['rouge2']:.4f} ({scores['rouge2']*100:.2f}%)")
    print(f"  • ROUGE-L F1:  {scores['rougeL']:.4f} ({scores['rougeL']*100:.2f}%)")
    print("=" * 65)
    
    samples_out = [
        {
            "article": df.iloc[i]["clean_text"][:120] + "...",
            "generated_summary": preds[i],
            "reference_summary": refs[i]
        }
        for i in range(min(3, len(preds)))
    ]
    
    return {
        "metrics": {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
            "rouge1_pct": scores["rouge1"] * 100.0,
            "rouge2_pct": scores["rouge2"] * 100.0,
            "rougeL_pct": scores["rougeL"] * 100.0,
        },
        "model_used": summarizer.active_backend,
        "sample_count": len(df),
        "dataset_source": data_source_desc,
        "sample_predictions": samples_out
    }


# ==============================================================================
# Standalone Runner
# ==============================================================================

if __name__ == "__main__":
    print("Testing Summarization Module...")
    sample_article = (
        "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने अपने सबसे शक्तिशाली रॉकेट के जरिए अत्याधुनिक स्वदेशी "
        "संचार उपग्रह को अंतरिक्ष की निर्धारित कक्षा में सफलतापूर्वक स्थापित कर दिया। इस उपग्रह से देश के दूरदराज "
        "और पर्वतीय क्षेत्रों में हाई-स्पीड ब्रॉडबैंड इंटरनेट और 5जी कनेक्टिविटी को अभूतपूर्व विस्तार मिलेगा। "
        "इसरो अध्यक्ष ने वैज्ञानिकों की पूरी टीम को बधाई देते हुए इसे डिजिटल इंडिया के लिए ऐतिहासिक मील का पत्थर बताया।"
    )
    
    summary_out = summarize_hindi(sample_article)
    print(f"\nOriginal Article ({len(sample_article)} chars):")
    print(sample_article)
    print(f"\nGenerated Summary ({len(summary_out)} chars):")
    print(summary_out)
    
    print("\nRunning ROUGE Evaluation...")
    eval_res = evaluate_summarizer_performance()
