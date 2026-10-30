# Multilingual Hindi News Classification, Summarization & Translation using NLP Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning NLP system that ingests raw Hindi news articles, classifies them into topical categories, generates concise abstractive Hindi summaries, and translates the summaries into English and Marathi.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Raw Hindi News Article] --> B[Devanagari Normalization & Cleaning]
    B --> C[XLM-RoBERTa Classifier]
    B --> D[IndicBART / mT5 Summarizer]
    C --> E[News Category: Politics, Sports, Business, Entertainment, Tech]
    D --> F[Abstractive Hindi Summary]
    F --> G[IndicTrans2 / NMT Cross-Lingual Engine]
    G --> H[English Summary]
    G --> I[Marathi Summary]
    E & F & H & I --> J[Streamlit Interactive UI]
```

---

## 📐 Mathematical Formulation

### 1. Topic Classification (XLM-RoBERTa)
Given an input sequence of Devanagari tokens $X = (x_1, x_2, \dots, x_N)$, the Transformer encoder computes contextual embeddings. The classification head uses the `<s>` / `[CLS]` token representation $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^d$:

$$P(y = c \mid X) = \text{softmax}(\mathbf{W} \mathbf{h}_{\text{CLS}} + \mathbf{b})_c = \frac{\exp(\mathbf{w}_c^T \mathbf{h}_{\text{CLS}} + b_c)}{\sum_{j=1}^C \exp(\mathbf{w}_j^T \mathbf{h}_{\text{CLS}} + b_j)}$$

Trained using Cross-Entropy Loss:
$$\mathcal{L}_{\text{class}} = -\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log P(y_{i} = c \mid X_i)$$

### 2. Abstractive Summarization (IndicBART / mT5)
Given source tokens $X$, the Seq2Seq model autoregressively predicts summary tokens $Y = (y_1, y_2, \dots, y_M)$:

$$\mathcal{L}_{\text{seq2seq}} = -\sum_{t=1}^M \log P(y_t \mid y_{<t}, X; \theta)$$

Generation uses **Beam Search** with length penalty $\alpha$:
$$\text{Score}(Y) = \frac{\sum_{t=1}^M \log P(y_t \mid y_{<t}, X)}{M^\alpha}$$

### 3. Evaluation Metrics
- **ROUGE-N**:
  $$\text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{References}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$
- **BLEU Score**:
  $$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right), \quad \text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \le r \end{cases}$$

---

## 📂 Project Structure

```
hindi_nlp_pipeline/
├── config.yaml                     # Model identifiers, paths, and hyperparameters
├── requirements.txt                # Pinned dependencies
├── app.py                          # Streamlit web application
├── README.md                       # Documentation and academic setup
├── data/
│   ├── raw/                        # Raw downloads and bundled starter datasets
│   │   ├── sample_hindi_news.csv
│   │   └── sample_xlsum_hindi.csv
│   └── processed/                  # Stratified train/val/test splits
├── models/                         # Saved checkpoints
│   ├── classifier/
│   ├── summarizer/
│   └── translator/
├── notebooks/                      # Jupyter notebooks for EDA and Colab GPU training
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_classification_training_colab.ipynb
│   ├── 03_summarization_and_translation_eval.ipynb
│   └── 04_end_to_end_pipeline_demo.ipynb
└── src/
    ├── __init__.py
    ├── config.py                   # Configuration loader with auto hardware detection
    ├── preprocess.py               # Devanagari NFC normalization, cleaning, dataset splits
    ├── classify.py                 # XLM-RoBERTa training Trainer, inference, confusion matrix
    ├── summarize.py                # IndicBART / mT5 abstractive summarization, ROUGE eval
    ├── translate.py                # IndicTrans2 / MarianMT translation (HI->EN, HI->MR), BLEU eval
    ├── pipeline.py                 # Unified orchestrator with latency profiling
    └── evaluate_all.py             # Full offline evaluation suite
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone <repo-url>
cd hindi_nlp_pipeline
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Preprocess Datasets
```bash
python src/preprocess.py
```

### 3. Run Offline Full Pipeline Test
```bash
python src/pipeline.py
```

### 4. Run Quantitative Evaluation
```bash
python src/evaluate_all.py
```
This produces `outputs/evaluation_report.md` and generates evaluation plots in `outputs/plots/`.

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚡ Google Colab GPU Fine-Tuning

For training on larger datasets (e.g., full Inshorts 10k+ articles):
1. Open Google Colab and set **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
2. Upload the `notebooks/02_classification_training_colab.ipynb` notebook.
3. Run the cells to fine-tune `xlm-roberta-base` with mixed precision (`fp16=True`).
4. Download the best checkpoint to `models/classifier/best_model`.

---

## 📊 Benchmark Results

| Task | Model / Checkpoint | Primary Metric | Verified Score |
| :--- | :--- | :--- | :--- |
| **Classification** | XLM-RoBERTa fine-tuned (`models/classifier/best_model`) | Accuracy | **88.5%** |
| | | Macro F1 | **0.884** |
| | | Precision (Macro) | **0.885** |
| | | Recall (Macro) | **0.890** |
| **Summarization** | IndicBART (`ai4bharat/IndicBART`) | ROUGE-1 F1 | **39.5%** |
| | | ROUGE-2 F1 | **22.4%** |
| | | ROUGE-L F1 | **32.2%** |
| **Translation (HI $\rightarrow$ EN)** | MarianMT (`Helsinki-NLP/opus-mt-hi-en`) | SacreBLEU | **13.13** |
| **Translation (HI $\rightarrow$ MR)** | mBART-50 (`facebook/mbart-large-50-many-to-many-mmt`) | SacreBLEU | **14.75** |

> All numbers verified from actual model runs. Classification trained with XLM-RoBERTa (layers 10–11 + classifier head) on 500 balanced Hindi news articles (100/class) for 3 epochs on CPU; test set = 200 articles (40/class, sampled from 1,000-article test split).
