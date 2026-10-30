import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
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

from src.config import CONFIG
from src.classify import evaluate_classifier_performance
from src.summarize import evaluate_summarizer_performance
from src.translate import evaluate_translation_performance


def generate_evaluation_plots(
    class_results: Dict[str, Any],
    sum_results: Dict[str, Any],
    trans_results: Dict[str, Any],
    output_dir: Path
):
    """
    Generate clean comparative charts for the evaluation report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summarization ROUGE Plot
    plt.figure(figsize=(7, 4.5))
    rouge_metrics = sum_results["metrics"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    values = [rouge_metrics["rouge1"] * 100, rouge_metrics["rouge2"] * 100, rouge_metrics["rougeL"] * 100]
    
    bars = plt.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55, edgecolor="black", alpha=0.85)
    plt.ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
    plt.title(f"Abstractive Summarization Performance ({sum_results.get('model_used', 'XL-Sum')})", fontsize=12, fontweight="bold")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    sum_plot_path = output_dir / "summarization_rouge_chart.png"
    plt.savefig(sum_plot_path, dpi=300)
    plt.close()
    
    # 2. Translation BLEU Plot
    plt.figure(figsize=(7, 4.5))
    t_labels = ["Hindi → English (BLEU)", "Hindi → Marathi (BLEU)"]
    t_values = [
        trans_results["hindi_to_english"]["sacrebleu"],
        trans_results["hindi_to_marathi"]["sacrebleu"]
    ]
    t_bars = plt.bar(t_labels, t_values, color=["#3b82f6", "#10b981"], width=0.45, edgecolor="black", alpha=0.85)
    plt.ylabel("SacreBLEU Score", fontsize=11, fontweight="bold")
    plt.title("Cross-Lingual Translation Quality (IndicTrans2 / NMT)", fontsize=12, fontweight="bold")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in t_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    trans_plot_path = output_dir / "translation_bleu_chart.png"
    plt.savefig(trans_plot_path, dpi=300)
    plt.close()

    # 3. Consolidated Multi-Panel Performance Dashboard Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: Classification F1 / Accuracy
    c_labels = ["Accuracy", "Precision", "Recall", "Macro F1"]
    c_vals = [
        class_results.get("accuracy", 0) * 100,
        class_results.get("precision_macro", 0) * 100,
        class_results.get("recall_macro", 0) * 100,
        class_results.get("f1_macro", 0) * 100,
    ]
    axes[0].bar(c_labels, c_vals, color=["#4f46e5", "#06b6d4", "#10b981", "#f59e0b"], edgecolor="black", alpha=0.9)
    axes[0].set_title("1. News Classification (XLM-RoBERTa)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for b in axes[0].patches:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel B: Summarization ROUGE
    s_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    s_vals = [rouge_metrics["rouge1"] * 100, rouge_metrics["rouge2"] * 100, rouge_metrics["rougeL"] * 100]
    axes[1].bar(s_labels, s_vals, color=["#3b82f6", "#ec4899", "#8b5cf6"], edgecolor="black", alpha=0.9)
    axes[1].set_title("2. Abstractive Summarization (ROUGE)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("F1 Score (%)", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in axes[1].patches:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel C: Translation SacreBLEU
    axes[2].bar(t_labels, t_values, color=["#0ea5e9", "#14b8a6"], edgecolor="black", alpha=0.9)
    axes[2].set_title("3. Multilingual Translation (SacreBLEU)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("SacreBLEU (0-100)", fontsize=10, fontweight="bold")
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for b in axes[2].patches:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.suptitle("Multilingual Hindi News Processing Pipeline - Quantitative Benchmark Dashboard", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    dashboard_path = output_dir / "pipeline_comprehensive_evaluation.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_consolidated_summary_table(
    class_results: Dict[str, Any],
    sum_results: Dict[str, Any],
    trans_results: Dict[str, Any]
):
    """
    Print clean ASCII metrics summary table to terminal.
    """
    acc = class_results.get("accuracy", 0.0) * 100
    p_macro = class_results.get("precision_macro", 0.0) * 100
    r_macro = class_results.get("recall_macro", 0.0) * 100
    f1_macro = class_results.get("f1_macro", 0.0)
    f1_wt = class_results.get("f1_weighted", 0.0)
    
    r1 = sum_results["metrics"]["rouge1"] * 100
    r2 = sum_results["metrics"]["rouge2"] * 100
    rl = sum_results["metrics"]["rougeL"] * 100
    
    en_sacre = trans_results["hindi_to_english"]["sacrebleu"]
    en_bleu = trans_results["hindi_to_english"]["bleu"]
    mr_sacre = trans_results["hindi_to_marathi"]["sacrebleu"]
    mr_bleu = trans_results["hindi_to_marathi"]["bleu"]
    
    print("\n" + "=" * 80)
    print("      MULTILINGUAL HINDI NEWS NLP PIPELINE — CONSOLIDATED EVALUATION REPORT")
    print("=" * 80)
    print(" 1. NEWS TOPIC CLASSIFICATION (5 Classes)")
    print(f"    • Model:                {class_results.get('model_loaded', 'XLM-RoBERTa')}")
    print(f"    • Type:                 {class_results.get('model_type', 'Fine-Tuned Checkpoint')}")
    print(f"    • Overall Accuracy:     {acc:.2f}%")
    print(f"    • Precision (Macro):    {p_macro:.2f}%")
    print(f"    • Recall (Macro):       {r_macro:.2f}%")
    print(f"    • Macro F1-Score:       {f1_macro:.4f}")
    print(f"    • Weighted F1-Score:    {f1_wt:.4f}")
    print("-" * 80)
    print(" 2. ABSTRACTIVE SUMMARIZATION (XL-Sum Hindi Benchmark)")
    print(f"    • Model / Backend:      {sum_results.get('model_used', 'IndicBART / mT5')}")
    print(f"    • Samples Evaluated:    {sum_results.get('sample_count', 0)}")
    print(f"    • ROUGE-1 F1:           {r1:.2f}% ({sum_results['metrics']['rouge1']:.4f})")
    print(f"    • ROUGE-2 F1:           {r2:.2f}% ({sum_results['metrics']['rouge2']:.4f})")
    print(f"    • ROUGE-L F1:           {rl:.2f}% ({sum_results['metrics']['rougeL']:.4f})")
    print("-" * 80)
    print(" 3. MULTILINGUAL TRANSLATION (Cross-Lingual NMT)")
    print(f"    • Samples Evaluated:    {trans_results.get('sample_count', 0)}")
    print(f"    • Hindi -> English:     SacreBLEU: {en_sacre:.2f} | Standard BLEU: {en_bleu:.4f}")
    print(f"    • Hindi -> Marathi:     SacreBLEU: {mr_sacre:.2f} | Standard BLEU: {mr_bleu:.4f}")
    print("=" * 80 + "\n")


def generate_markdown_report(
    class_results: Dict[str, Any],
    sum_results: Dict[str, Any],
    trans_results: Dict[str, Any],
    output_file: Path
):
    """
    Generate comprehensive evaluation markdown document.
    """
    rouge = sum_results["metrics"]
    hi_en = trans_results["hindi_to_english"]
    hi_mr = trans_results["hindi_to_marathi"]
    
    md_content = f"""# Academic Evaluation Report: Multilingual Hindi News NLP System

## Executive Summary
This report documents the rigorous quantitative evaluation of the three core pipeline components:
1. **News Topic Classification** (*XLM-RoBERTa / IndicBERT*)
2. **Abstractive News Summarization** (*IndicBART / mT5-XLSum*)
3. **Multilingual Translation** (*IndicTrans2 / NMT*)

---

## 1. Classification Performance (5 Classes)
- **Target Classes**: Politics, Sports, Business, Entertainment, Technology
- **Model Checkpoint**: `{class_results.get('model_loaded', 'models/classifier/best_model')}`
- **Model Status**: `{class_results.get('model_type', 'Fine-Tuned Checkpoint')}`
- **Overall Accuracy**: **{class_results.get('accuracy', 0.0) * 100:.2f}%**
- **Macro Precision**: **{class_results.get('precision_macro', 0.0) * 100:.2f}%**
- **Macro Recall**: **{class_results.get('recall_macro', 0.0) * 100:.2f}%**
- **Macro F1-Score**: **{class_results.get('f1_macro', 0.0):.4f}**
- **Weighted F1-Score**: **{class_results.get('f1_weighted', 0.0):.4f}**

### Per-Class Detailed Breakdown
```
{class_results.get('report', 'Report unavailable')}
```

---

## 2. Abstractive Summarization Performance (XL-Sum Hindi Benchmark)
Abstractive Hindi summarization was measured against gold-standard reference summaries using the standard ROUGE metric suite:
- **Backend Model**: `{sum_results.get('model_used', 'IndicBART / mT5')}`
- **Dataset Source**: `{sum_results.get('dataset_source', 'XL-Sum Hindi')}`
- **Samples Evaluated**: `{sum_results.get('sample_count', 0)}`

| Metric | F1 Score | Percentage |
| :--- | :--- | :--- |
| **ROUGE-1** (Unigram Overlap) | `{rouge['rouge1']:.4f}` | **{rouge['rouge1']*100:.2f}%** |
| **ROUGE-2** (Bigram Overlap) | `{rouge['rouge2']:.4f}` | **{rouge['rouge2']*100:.2f}%** |
| **ROUGE-L** (Longest Common Subsequence) | `{rouge['rougeL']:.4f}` | **{rouge['rougeL']*100:.2f}%** |

---

## 3. Multilingual Translation Performance
Evaluated across Hindi → English and Hindi → Marathi translation directions using BLEU and SacreBLEU:
- **Samples Evaluated**: `{trans_results.get('sample_count', 0)}`

| Language Direction | BLEU Score | SacreBLEU Score |
| :--- | :--- | :--- |
| **Hindi $\\rightarrow$ English** | `{hi_en['bleu']:.4f}` | **{hi_en['sacrebleu']:.2f}** |
| **Hindi $\\rightarrow$ Marathi** | `{hi_mr['bleu']:.4f}` | **{hi_mr['sacrebleu']:.2f}** |

---

## 4. Qualitative Samples

### Sample 1: Hindi Summary $\\rightarrow$ Multilingual Output
- **Hindi Original Summary**:
  > {trans_results['samples'][0]['hindi_source']}
- **English Translation**:
  > {trans_results['samples'][0]['english_prediction']}
- **Marathi Translation**:
  > {trans_results['samples'][0]['marathi_prediction']}

---
*Report generated automatically by the Multilingual Hindi NLP Evaluation Suite.*
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"[Evaluation Report] Saved markdown report to: {output_file}")


def run_full_evaluation(
    test_classify_csv: Optional[str] = None,
    test_sum_csv: Optional[str] = None,
    test_trans_csv: Optional[str] = None,
    classify_model_path: Optional[str] = None,
    sum_samples: Optional[int] = 100,
    trans_samples: Optional[int] = 100
) -> Dict[str, Any]:
    """
    Execute complete evaluation workflow across all three pipeline components.
    """
    print("=" * 80)
    print("STARTING COMPREHENSIVE MULTILINGUAL HINDI NLP EVALUATION")
    print("=" * 80)
    
    # 1. Classification Evaluation
    print("\n[Stage 1/3] Evaluating News Topic Classifier (XLM-RoBERTa)...")
    class_res = evaluate_classifier_performance(
        test_csv=test_classify_csv,
        model_path=classify_model_path,
        save_plot=True
    )
    
    # 2. Summarization Evaluation
    print("\n[Stage 2/3] Evaluating Hindi Abstractive Summarizer (XL-Sum Benchmark)...")
    sum_res = evaluate_summarizer_performance(
        test_csv=test_sum_csv,
        sample_size=sum_samples
    )
    
    # 3. Translation Evaluation
    print("\n[Stage 3/3] Evaluating Hindi->English & Hindi->Marathi Translation...")
    trans_res = evaluate_translation_performance(
        test_csv=test_trans_csv,
        sample_size=trans_samples
    )
    
    # 4. Plots, Summary JSON, and Markdown Report
    print("\n[Stage 4/4] Compiling Plots and Consolidated Evaluation Documents...")
    plots_dir = CONFIG.outputs_dir / "plots"
    generate_evaluation_plots(class_res, sum_res, trans_res, plots_dir)
    
    report_file = CONFIG.outputs_dir / "evaluation_report.md"
    generate_markdown_report(class_res, sum_res, trans_res, report_file)
    
    # Save machine-readable JSON metrics
    summary_json = {
        "classification": {
            "model_path": class_res.get("model_loaded"),
            "model_type": class_res.get("model_type"),
            "accuracy": class_res.get("accuracy"),
            "precision_macro": class_res.get("precision_macro"),
            "recall_macro": class_res.get("recall_macro"),
            "f1_macro": class_res.get("f1_macro"),
            "precision_weighted": class_res.get("precision_weighted"),
            "recall_weighted": class_res.get("recall_weighted"),
            "f1_weighted": class_res.get("f1_weighted"),
        },
        "summarization": {
            "model_used": sum_res.get("model_used"),
            "sample_count": sum_res.get("sample_count"),
            "rouge1": sum_res["metrics"]["rouge1"],
            "rouge2": sum_res["metrics"]["rouge2"],
            "rougeL": sum_res["metrics"]["rougeL"],
        },
        "translation": {
            "hindi_to_english_sacrebleu": trans_res["hindi_to_english"]["sacrebleu"],
            "hindi_to_english_bleu": trans_res["hindi_to_english"]["bleu"],
            "hindi_to_marathi_sacrebleu": trans_res["hindi_to_marathi"]["sacrebleu"],
            "hindi_to_marathi_bleu": trans_res["hindi_to_marathi"]["bleu"],
        }
    }
    json_path = CONFIG.outputs_dir / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[Evaluation Summary] Saved JSON metrics to: {json_path}")
    
    # 5. Print Consolidated ASCII Table
    print_consolidated_summary_table(class_res, sum_res, trans_res)
    
    print("=" * 80)
    print("ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"Report Markdown: {report_file}")
    print(f"Metrics JSON:    {json_path}")
    print(f"Plots Directory: {plots_dir}")
    print("=" * 80)
    
    return {
        "classification": class_res,
        "summarization": sum_res,
        "translation": trans_res,
        "summary": summary_json,
        "report_markdown": str(report_file),
        "summary_json": str(json_path)
    }


if __name__ == "__main__":
    run_full_evaluation()
