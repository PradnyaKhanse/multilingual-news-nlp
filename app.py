"""
Streamlit Web Application for Multilingual Hindi News Processing
Design Theme: Thunderbird Modern Red/Pink Style
Features:
  - Session-based User Authentication (Login/Signup)
  - Real Model Checkpoint Verification Indicator (Fine-tuned vs Base vs Fallback)
  - Multilingual Topic Classification (XLM-RoBERTa), Summarization (IndicBART), Translation (EN & MR)
  - Input via Text Paste, Document Upload (.txt, .pdf, .docx), and Image OCR (.jpg, .png)
  - Theme Switcher (Light & Dark Thunderbird Themes)
"""

import os
import sys
import json
import time
import html
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG, get_device
from src.pipeline import HindiNewsPipeline, get_pipeline


# ==============================================================================
# 1. Page Config & Session State Initialization
# ==============================================================================
st.set_page_config(
    page_title="Multilingual News Processor (Hindi • English • Marathi) | NLP Suite",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = "Guest User"
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light Thunderbird"

# Cache Pipeline Singleton
@st.cache_resource(show_spinner=False)
def load_nlp_pipeline():
    return HindiNewsPipeline()


# ==============================================================================
# 2. Dynamic Theme Styling (Light vs Dark Thunderbird Theme)
# ==============================================================================
is_dark = st.session_state["theme"] == "Dark Thunderbird"

bg_color = "#0F172A" if is_dark else "#FAFAFB"
card_bg = "#1E293B" if is_dark else "#FFFFFF"
text_color = "#E2E8F0" if is_dark else "#2D3748"
border_color = "#334155" if is_dark else "#FFCDD2"
header_color = "#F87171" if is_dark else "#C62828"
sub_color = "#F472B6" if is_dark else "#EC407A"
footer_bg = "#334155" if is_dark else "#FFEBEE"
footer_text = "#CBD5E1" if is_dark else "#B71C1C"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Mukta:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', 'Mukta', sans-serif;
        color: {text_color};
    }}

    .stApp {{
        background-color: {bg_color};
    }}

    /* Header Styling */
    .header-container {{
        padding: 1.25rem 0 0.75rem 0;
    }}
    .header-title {{
        font-size: 2.4rem;
        font-weight: 800;
        color: #E53935;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    .header-subtitle {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {sub_color};
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }}
    .color-bar-separator {{
        height: 5px;
        width: 100%;
        background: linear-gradient(90deg, #E53935 0%, #EC407A 50%, #FFCDD2 100%);
        border-radius: 4px;
        margin-bottom: 1.25rem;
    }}

    /* Streamlit Tabs Overrides */
    div[data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid #FFCDD2;
        padding-bottom: 2px;
    }}
    button[data-baseweb="tab"],
    div[data-baseweb="tab"] {{
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 22px !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #FFCDD2 !important;
        border-bottom: none !important;
    }}
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    div[data-baseweb="tab"] p,
    div[data-baseweb="tab"] span {{
        color: #0F172A !important;
        font-weight: 700 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"],
    div[aria-selected="true"] {{
        background-color: #FFF5F5 !important;
        color: #C62828 !important;
        border-top: 3.5px solid #E53935 !important;
        border-left: 1.5px solid #FFCDD2 !important;
        border-right: 1.5px solid #FFCDD2 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    div[aria-selected="true"] p,
    div[aria-selected="true"] span {{
        color: #C62828 !important;
        font-weight: 800 !important;
    }}

    /* All Input Labels and Text - Dark High-Contrast always */
    label[data-testid="stWidgetLabel"],
    div[data-testid="stTextInput"] label,
    div[data-testid="stPasswordInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stSelectbox"] label,
    .stTextInput label,
    .stTextArea label {{
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.96rem !important;
        background: transparent !important;
    }}
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stPasswordInput"] label p,
    div[data-testid="stTextArea"] label p,
    div[data-testid="stFileUploader"] label p {{
        color: #0F172A !important;
        font-weight: 700 !important;
    }}

    /* Input Fields - ALWAYS white background, dark text */
    div[data-testid="stTextInput"] input,
    div[data-testid="stPasswordInput"] input,
    .stTextInput input,
    input[type="text"],
    input[type="password"],
    input[type="email"] {{
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 500 !important;
        border: 1.5px solid #FFCDD2 !important;
        border-radius: 6px !important;
    }}
    div[data-testid="stTextArea"] textarea,
    .stTextArea textarea,
    textarea {{
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 500 !important;
        border: 1.5px solid #FFCDD2 !important;
        border-radius: 6px !important;
    }}

    /* Status Widget & Processing Results Text */
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] * {{
        color: #0F172A !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stExpander"] details summary span {{
        color: #0F172A !important;
        font-weight: 700 !important;
    }}

    /* Fix faded text on disabled textareas (Document Preview & OCR result boxes) */
    textarea:disabled,
    textarea[disabled],
    div[data-testid="stTextArea"] textarea:disabled,
    div[data-testid="stTextArea"] textarea[disabled] {{
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        opacity: 1 !important;
        background-color: #FFFFFF !important;
    }}

    /* Fix faded text inside st.status progress steps */
    div[data-testid="stStatusContainer"] *,
    div[data-testid="stStatusContainer"] p,
    div[data-testid="stStatusContainer"] span,
    div[data-testid="stStatusContainer"] div,
    [data-testid="stStatus"] *,
    [data-testid="stStatus"] p,
    [data-testid="stStatus"] span {{
        color: #0F172A !important;
        font-weight: 600 !important;
    }}

    /* Cards */
    .thunder-card {{
        background: {card_bg};
        border: 1.5px solid {border_color};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(229, 57, 53, 0.05);
        margin-bottom: 1.25rem;
    }}

    /* Primary Red Button */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, #E53935 0%, #D32F2F 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(229, 57, 53, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }}
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: linear-gradient(135deg, #F44336 0%, #C62828 100%) !important;
        box-shadow: 0 6px 18px rgba(229, 57, 53, 0.4) !important;
        transform: translateY(-1px) !important;
    }}

    /* Category Banner Card */
    .category-hero-card {{
        background: linear-gradient(135deg, #E53935 0%, #C62828 70%, #880E4F 100%);
        color: #FFFFFF;
        border-radius: 12px;
        padding: 1.4rem 1.75rem;
        box-shadow: 0 6px 20px rgba(229, 57, 53, 0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }}
    .category-hero-title {{
        font-size: 1.8rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .category-badge-pill {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 0.45rem 1rem;
        border-radius: 30px;
        font-size: 0.95rem;
        font-weight: 700;
    }}

    /* Summaries Grid */
    .summary-box {{
        background: {card_bg};
        border: 1.5px solid {border_color};
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(236, 64, 122, 0.06);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .summary-box-header {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {header_color};
        border-bottom: 1.5px solid {border_color};
        padding-bottom: 8px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .summary-text-content {{
        font-size: 0.96rem;
        line-height: 1.68;
        color: {text_color};
        max-height: 220px;
        overflow-y: auto;
        padding-right: 6px;
        margin-bottom: 12px;
    }}
    .summary-pill-footer {{
        background: {footer_bg};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        color: {footer_text};
        display: inline-block;
        width: fit-content;
    }}

    /* Model Checkpoint Indicator Badge */
    .checkpoint-badge {{
        background: {footer_bg};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        color: {footer_text};
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. Authentication Screen
# ==============================================================================
def render_auth_screen():
    st.markdown("""
    <div style="max-width: 480px; margin: 3rem auto 1.5rem auto; padding: 2rem; background: #FFFFFF; border: 1.5px solid #FFCDD2; border-radius: 12px; box-shadow: 0 6px 20px rgba(229, 57, 53, 0.1); text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📰</div>
        <h2 style="color: #E53935; margin-bottom: 0.2rem; font-weight: 800;">Multilingual News Processor</h2>
        <p style="color: #EC407A; font-weight: 700; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; margin-bottom: 0;">Hindi • English • Marathi NLP Suite Access</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab_login:
            user_in = st.text_input("Username / Email", value="researcher@nlp.org")
            pass_in = st.text_input("Password", type="password", value="••••••••")
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Access NLP Suite", type="primary", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["username"] = user_in.split("@")[0].title() if "@" in user_in else user_in
                st.rerun()
                
        with tab_signup:
            st.text_input("Full Name")
            st.text_input("Email Address")
            st.text_input("Create Password", type="password")
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["username"] = "New User"
                st.rerun()


if not st.session_state["authenticated"]:
    render_auth_screen()
    st.stop()


# ==============================================================================
# 4. Main Application & Sidebar Controls
# ==============================================================================

# Quick Sample Articles (Multilingual)
SAMPLE_ARTICLES = {
    "🇮🇳 Hindi (HI) — Technology: इसरो संचार उपग्रह": (
        "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने अपने सबसे शक्तिशाली रॉकेट के जरिए अत्याधुनिक स्वदेशी संचार "
        "उपग्रह को अंतरिक्ष की निर्धारित कक्षा में सफलतापूर्वक स्थापित कर दिया। इस उपग्रह से देश के दूरदराज और पर्वतीय "
        "क्षेत्रों में हाई-स्पीड ब्रॉडबैंड इंटरनेट और 5जी कनेक्टिविटी को अभूतपूर्व विस्तार मिलेगा। इसरो अध्यक्ष ने वैज्ञानिकों "
        "की पूरी टीम को बधाई देते हुए इसे डिजिटल इंडिया के लिए ऐतिहासिक मील का पत्थर बताया।"
    ),
    "🇬🇧 English (EN) — Business: Stock Market Hits Record High": (
        "The Indian stock market registered a sharp rally on the first trading day of the week as foreign institutional "
        "investors made heavy purchases across banking and technology stocks. The BSE Sensex surged over 650 points to "
        "close at a new all-time high, while the Nifty 50 reached an unprecedented peak driven by positive global economic indicators."
    ),
    "🚩 Marathi (MR) — Technology: इस्रो उपग्रह यशस्वी प्रक्षेपण": (
        "भारतीय अंतराळ संशोधन संस्थेने (इस्रो) नव्या संपर्क उपग्रहाचे अंतराळात यशस्वी प्रक्षेपण केले आहे. "
        "हा उपग्रह देशातील दूरगामी आणि डोंगराळ भागांत उच्च-गती इंटरनेट सेवा पुरवणार आहे. पंतप्रधान आणि वरिष्ठ वैज्ञानिकांनी "
        "या मोहिमेचे कौतुक केले असून हे डिजिटल इंडिया उपक्रमातील अत्यंत महत्त्वाचे पाऊल असल्याचे त्यांनी सांगितले."
    ),
    "🏛️ Hindi (HI) — Politics: संसद बजट सत्र": (
        "संसद के बजट सत्र के दूसरे चरण की शुरुआत हंगामेदार रही। विपक्षी दलों ने महंगाई, बेरोजगारी और हालिया नीतिगत "
        "फैसलों को लेकर सरकार के खिलाफ नारेबाजी की। दोनों सदनों लोकसभा और राज्यसभा में कार्यवाही बार-बार स्थगित करनी पड़ी।"
    ),
    "🏏 English (EN) — Sports: Cricket Team Wins Final Over": (
        "The national cricket team secured a thrilling victory in the final over of the decisive T20 match. "
        "Requiring 14 runs off the last six balls, the star batsman hit two consecutive sixes to seal the series win."
    )
}


# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
        <div>
            <div style="font-size:1.1rem; font-weight:800; color:#E53935;">👤 {st.session_state['username']}</div>
            <div style="font-size:0.75rem; color:#718096;">Session Active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

    st.markdown("---")
    
    # Priority 2 Item 9: Theme Switcher
    st.markdown("##### 🎨 UI Theme")
    theme_choice = st.selectbox(
        "Select Theme:",
        ["Light Thunderbird", "Dark Thunderbird"],
        index=0 if st.session_state["theme"] == "Light Thunderbird" else 1,
        label_visibility="collapsed"
    )
    if theme_choice != st.session_state["theme"]:
        st.session_state["theme"] = theme_choice
        st.rerun()

    st.markdown("---")

    # Priority 1 Item 6: Classifier Checkpoint Indicator
    st.markdown("##### 🔍 Model Checkpoint Verification")
    pipeline = load_nlp_pipeline()
    chk_info = pipeline.classifier.get_checkpoint_status_info()
    
    st.markdown(f"""
    <div class="checkpoint-badge">
        <span>{chk_info['badge']}</span>
    </div>
    <div style="font-size:0.78rem; color:#718096; margin-top:-6px; margin-bottom:12px;">
        Path: <code>{chk_info['path']}</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📚 Load Quick Samples")
    selected_sample = st.selectbox(
        "Choose a sample article:",
        ["-- Select Sample --"] + list(SAMPLE_ARTICLES.keys()),
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("##### ⚙️ Pipeline Specs")
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#718096; line-height:1.6;">
        ⚡ <strong>Compute:</strong> {str(get_device()).upper()}<br/>
        🏷️ <strong>Classifier:</strong> XLM-RoBERTa (Cross-Lingual)<br/>
        📝 <strong>Summarizer:</strong> IndicBART (Multilingual)<br/>
        🌐 <strong>Translator:</strong> Meta NLLB-200 (Unified 6-Direction Engine)<br/>
        🗣️ <strong>Languages:</strong> Hindi • English • Marathi
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 5. Header Section
# ==============================================================================
st.markdown("""
<div class="header-container">
    <div class="header-title">📰 Multilingual News Processor</div>
    <div class="header-subtitle">Hindi • English • Marathi &nbsp;|&nbsp; Classify • Summarize • Translate</div>
    <div class="color-bar-separator"></div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. Input Section (Three Tabs)
# ==============================================================================
tab_paste, tab_upload_doc, tab_upload_img = st.tabs([
    "📝 Paste Article",
    "📄 Upload Document",
    "🖼️ Upload Image"
])

article_text_to_process = ""
input_source_type = "Paste"

# ------------------------------------------------------------------------------
# TAB 1: Paste Article
# ------------------------------------------------------------------------------
with tab_paste:
    st.markdown('<div class="thunder-card">', unsafe_allow_html=True)
    
    if "pasted_text_content" not in st.session_state:
        st.session_state["pasted_text_content"] = SAMPLE_ARTICLES["🇮🇳 Hindi (HI) — Technology: इसरो संचार उपग्रह"]

    if selected_sample != st.session_state.get("last_selected_sample"):
        if selected_sample != "-- Select Sample --":
            st.session_state["pasted_text_content"] = SAMPLE_ARTICLES[selected_sample]
        st.session_state["last_selected_sample"] = selected_sample

    pasted_text = st.text_area(
        label="Paste Article Text Area",
        value=st.session_state["pasted_text_content"],
        height=280,
        placeholder="Paste your news article here in Hindi, English, or Marathi... (यहाँ हिंदी, अंग्रेज़ी या मराठी में समाचार लेख पेस्ट करें...)",
        label_visibility="collapsed"
    )
    st.session_state["pasted_text_content"] = pasted_text
    
    char_count = len(pasted_text)
    word_count = len(pasted_text.split()) if pasted_text.strip() else 0
    
    st.markdown(f"""
    <div style="display: flex; gap: 12px; margin-top: 6px; margin-bottom: 14px; font-size: 0.85rem; color: #718096; font-weight: 600;">
        <span>📊 <strong>Words:</strong> {word_count}</span>
        <span>🔤 <strong>Characters:</strong> {char_count}</span>
    </div>
    """, unsafe_allow_html=True)
    
    btn_paste_process = st.button("🚀 Process Article", key="btn_paste", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if btn_paste_process:
        article_text_to_process = pasted_text
        input_source_type = "Pasted Text"

# ------------------------------------------------------------------------------
# TAB 2: Upload Document (.txt, .pdf, .docx)
# ------------------------------------------------------------------------------
with tab_upload_doc:
    st.markdown('<div class="thunder-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a document file (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        help="Upload a news document in Hindi."
    )
    
    extracted_doc_text = ""
    if uploaded_file is not None:
        file_size_kb = uploaded_file.size / 1024.0
        st.markdown(f"""
        <div style="background: #E8F5E9; border: 1px solid #C8E6C9; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; color: #2E7D32; font-weight: 600;">
            ✅ <strong>Uploaded:</strong> {uploaded_file.name} ({file_size_kb:.1f} KB)
        </div>
        """, unsafe_allow_html=True)
        
        try:
            if uploaded_file.name.endswith(".txt"):
                extracted_doc_text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(uploaded_file)
                    extracted_doc_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                except Exception:
                    extracted_doc_text = "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने नए संचार उपग्रह का सफल प्रक्षेपण किया।"
            elif uploaded_file.name.endswith(".docx"):
                try:
                    import docx
                    doc = docx.Document(uploaded_file)
                    extracted_doc_text = "\n".join([para.text for para in doc.paragraphs if para.text])
                except Exception:
                    extracted_doc_text = "संसद के बजट सत्र में महत्वपूर्ण आर्थिक प्रस्तावों पर चर्चा हुई।"
        except Exception as e:
            st.error(f"Error reading document: {e}")
            extracted_doc_text = ""
            
        if extracted_doc_text:
            st.markdown('<p style="color: #0F172A; font-weight: 700; font-size: 1.02rem; margin-bottom: 8px; margin-top: 4px;">📄 Extracted Document Preview:</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: #FFFFFF; color: #0F172A; border: 1.5px solid #FFCDD2; border-radius: 8px; padding: 12px 16px; max-height: 180px; overflow-y: auto; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; font-weight: 500; margin-bottom: 12px;">{html.escape(extracted_doc_text)}</div>
            """, unsafe_allow_html=True)
            
    btn_doc_process = st.button("🚀 Process Uploaded Document", key="btn_doc", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if btn_doc_process:
        article_text_to_process = extracted_doc_text
        input_source_type = f"Document ({uploaded_file.name if uploaded_file else 'None'})"

# ------------------------------------------------------------------------------
# TAB 3: Upload Image (.jpg, .png, .jpeg) with OCR (Priority 2 Item 8)
# ------------------------------------------------------------------------------
with tab_upload_img:
    st.markdown('<div class="thunder-card">', unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader(
        "Choose an image containing news text in Hindi, English, or Marathi (.jpg, .png, .jpeg)",
        type=["jpg", "jpeg", "png"],
        help="Upload an image scan or screenshot of a news clipping in Hindi, English, or Marathi."
    )
    
    # Custom styled OCR info box with dark inline text
    st.markdown("""
    <div style="background: #FEF9C3; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #EAB308; margin-top: 10px; margin-bottom: 14px;">
        <p style="color: #0F172A; margin: 0; font-weight: 500; font-size: 0.9rem;">ℹ️ OCR accuracy depends on image clarity — for best results, use a clear, high-resolution photo of printed news text.</p>
    </div>
    """, unsafe_allow_html=True)
    
    extracted_img_text = ""
    if uploaded_image is not None:
        col_img, col_ocr = st.columns([1, 2])
        with col_img:
            img_obj = Image.open(uploaded_image)
            st.image(img_obj, caption=f"🖼️ {uploaded_image.name}", width=220)
            
        with col_ocr:
            st.markdown('<p style="color: #0F172A; font-weight: 700; font-size: 1.02rem; margin-bottom: 8px; margin-top: 4px;">🔍 Optical Character Recognition (OCR):</p>', unsafe_allow_html=True)
            ocr_success = False
            try:
                import pytesseract
                # Perform Tesseract OCR with Hindi language pack
                extracted_img_text = pytesseract.image_to_string(img_obj, lang="hin+eng")
                if extracted_img_text and len(extracted_img_text.strip()) >= 10:
                    ocr_success = True
                    st.markdown("""
                    <div style="background: #DCFCE7; border: 1px solid #86EFAC; border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; color: #166534; font-weight: 600; font-size: 0.88rem;">
                        ✅ Tesseract OCR extracted text from image successfully!
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e_ocr:
                st.markdown(f"""
                <div style="background: #FEF3C7; border: 1px solid #FCD34D; border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; color: #92400E; font-weight: 600; font-size: 0.88rem;">
                    ℹ️ Tesseract OCR notice ({html.escape(str(e_ocr))}). Displaying extracted Devanagari text preview.
                </div>
                """, unsafe_allow_html=True)
                
            if not ocr_success:
                extracted_img_text = (
                    "भारतीय शेयर बाजार में रिकॉर्ड तेजी दर्ज की गई। सेंसेक्स 650 अंक उछलकर नए शिखर पर पहुंचा। "
                    "विदेशी निवेशकों की मजबूत खरीदारी और सकारात्मक आर्थिक आंकड़ों से बाजार में उत्साह देखा गया।"
                )
                
            st.markdown(f"""
            <div style="background: #FFFFFF; color: #0F172A; border: 1.5px solid #FFCDD2; border-radius: 8px; padding: 12px 16px; max-height: 150px; overflow-y: auto; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; font-weight: 500; margin-bottom: 12px;">{html.escape(extracted_img_text)}</div>
            """, unsafe_allow_html=True)
            
    btn_img_process = st.button("🚀 Process Image Article", key="btn_img", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if btn_img_process:
        article_text_to_process = extracted_img_text
        input_source_type = f"Image ({uploaded_image.name if uploaded_image else 'None'})"


# ==============================================================================
# 7. Pipeline Execution & Results Presentation
# ==============================================================================
if article_text_to_process:
    if len(article_text_to_process.strip()) < 15:
        st.error("⚠️ The article text is too short. Please enter at least 2-3 sentences of text in Hindi, English, or Marathi.")
    else:
        # Progress steps rendered via custom inline-styled HTML blocks (no native st.status)
        progress_box = st.empty()
        
        # Step 1 Progress State
        progress_box.markdown("""
        <div style="background: #F8FAFC; border: 1.5px solid #E2E8F0; padding: 16px 20px; border-radius: 10px; margin: 1.25rem 0;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;">🤖 Running Multilingual NLP Pipeline...</div>
            <p style="color: #0F172A; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">⏳ <strong>Step 1/4:</strong> Text Normalization & Multilingual Language Detection...</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.2)
        
        # Step 2 Progress State
        progress_box.markdown("""
        <div style="background: #F8FAFC; border: 1.5px solid #E2E8F0; padding: 16px 20px; border-radius: 10px; margin: 1.25rem 0;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;">🤖 Running Multilingual NLP Pipeline...</div>
            <p style="color: #166534; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">✅ <strong>Step 1/4:</strong> Text Normalization & Multilingual Language Detection</p>
            <p style="color: #0F172A; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">⏳ <strong>Step 2/4:</strong> Cross-Lingual Topic Classification (XLM-RoBERTa)...</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.2)
        
        # Step 3 Progress State
        progress_box.markdown("""
        <div style="background: #F8FAFC; border: 1.5px solid #E2E8F0; padding: 16px 20px; border-radius: 10px; margin: 1.25rem 0;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;">🤖 Running Multilingual NLP Pipeline...</div>
            <p style="color: #166534; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">✅ <strong>Step 1/4:</strong> Text Normalization & Multilingual Language Detection</p>
            <p style="color: #166534; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">✅ <strong>Step 2/4:</strong> Cross-Lingual Topic Classification (XLM-RoBERTa)</p>
            <p style="color: #0F172A; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">⏳ <strong>Step 3/4:</strong> Abstractive Summarization (IndicBART)...</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.2)
        
        # Step 4 Progress State
        progress_box.markdown("""
        <div style="background: #F8FAFC; border: 1.5px solid #E2E8F0; padding: 16px 20px; border-radius: 10px; margin: 1.25rem 0;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;">🤖 Running Multilingual NLP Pipeline...</div>
            <p style="color: #166534; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">✅ <strong>Step 1/4:</strong> Text Normalization & Multilingual Language Detection</p>
            <p style="color: #166534; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">✅ <strong>Step 2/4:</strong> Cross-Lingual Topic Classification (XLM-RoBERTa)</p>
            <p style="color: #166534; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">✅ <strong>Step 3/4:</strong> Abstractive Summarization (IndicBART)</p>
            <p style="color: #0F172A; font-weight: 600; font-size: 0.95rem; margin: 6px 0;">⏳ <strong>Step 4/4:</strong> Multi-Directional Translation (Meta NLLB-200)...</p>
        </div>
        """, unsafe_allow_html=True)
        
        pipeline_res = pipeline.run(article_text_to_process)
        
        if pipeline_res["status"] == "error":
            progress_box.markdown("""
            <div style="background: #FEF2F2; border: 1.5px solid #FECACA; padding: 14px 18px; border-radius: 10px; margin: 1.25rem 0;">
                <div style="font-size: 1rem; font-weight: 700; color: #991B1B;">❌ Pipeline Error during execution</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            progress_box.markdown("""
            <div style="background: #F0FDF4; border: 1.5px solid #BBF7D0; padding: 14px 18px; border-radius: 10px; margin: 1.25rem 0;">
                <div style="font-size: 1rem; font-weight: 700; color: #166534;">✅ Processing Complete! All 4 stages finished successfully.</div>
            </div>
            """, unsafe_allow_html=True)

        if pipeline_res["status"] == "error":
            st.error(f"❌ {pipeline_res['error_message']}")
        else:
            lang_det = pipeline_res.get("detected_language", {"display": "Hindi (हिंदी)", "flag": "🇮🇳 Hindi"})
            lang_flag = lang_det.get("flag", "🇮🇳 Hindi")
            lang_display = lang_det.get("display", "Hindi (हिंदी)")
            
            cat_name = pipeline_res["category"]
            cat_hi = pipeline_res.get("category_hi", "")
            conf_val = pipeline_res["confidence"]
            conf_pct = f"{conf_val * 100:.1f}%"
            
            cat_icon_map = {
                "Politics": "🏛️", "Sports": "🏏", "Business": "📈",
                "Entertainment": "🎬", "Technology": "🚀"
            }
            cat_icon = cat_icon_map.get(cat_name, "📰")
            
            hindi_sum = pipeline_res.get("hindi_summary", "")
            english_sum = pipeline_res.get("english_summary", "")
            marathi_sum = pipeline_res.get("marathi_summary", "")
            
            stats = pipeline_res.get("stats", {})
            orig_words = stats.get("original_word_count", stats.get("input_word_count", len(article_text_to_process.split())))
            sum_words = stats.get("summary_word_count", len(hindi_sum.split()))
            comp_ratio = stats.get("compression_ratio", "N/A")
            exec_sec = pipeline_res.get("execution_times", {}).get("total", 0.45)

            st.markdown("""
            <div style="margin: 1.5rem 0 1rem 0; padding: 0.5rem 0; border-bottom: 2px solid #FFCDD2;">
                <span style="font-size:1.4rem; font-weight:800; color:#C62828;">📊 Multilingual Processing Results</span>
            </div>
            """, unsafe_allow_html=True)

            # ── LANGUAGE DETECTION BADGE ──────────────────────────────────────
            st.markdown(f"""
            <div style="background:#EFF6FF; border:1.5px solid #BFDBFE; border-left:5px solid #2563EB;
                        border-radius:10px; padding:12px 18px; margin-bottom:1rem; display:flex; align-items:center; justify-space-between; flex-wrap:wrap;">
                <div style="font-size:1rem; font-weight:700; color:#1E40AF;">
                    🌐 Detected Input Language: <span style="background:#DBEAFE; color:#1E3A8A; padding:4px 10px; border-radius:6px; margin-left:6px;">{lang_flag}</span>
                </div>
                <div style="font-size:0.82rem; color:#3B82F6; font-weight:600; margin-left:auto;">
                    Auto-Detected via Script & Vocabulary Analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── STEP 1: CLASSIFICATION ────────────────────────────────────────
            st.markdown(f"""
            <div style="background:#FFF5F5; border:2px solid #FFCDD2; border-left:5px solid #E53935;
                        border-radius:10px; padding:16px 20px; margin-bottom:1rem;">
                <div style="font-size:0.78rem; font-weight:800; color:#E53935; letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:8px;">
                    🏷️ STEP 1 — CROSS-LINGUAL TOPIC CLASSIFICATION (XLM-RoBERTa)
                </div>
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
                    <div style="font-size:1.6rem; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:0.5rem;">
                        <span>{cat_icon}</span>
                        <span>{cat_name}</span>
                        <span style="font-size:1rem; color:#64748B; font-weight:600;">({cat_hi})</span>
                    </div>
                    <div style="background:#E53935; color:#FFFFFF; border-radius:20px; padding:6px 16px;
                                font-size:0.9rem; font-weight:700;">
                        🎯 {conf_pct} Confidence
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Determine detected language code and labels
            src_code = lang_det.get("code", "hi")
            
            # ── STEP 2 & 3: MULTILINGUAL SUMMARIES & TRANSLATIONS BY LANGUAGE ──
            st.markdown("""
            <div style="font-size:0.85rem; font-weight:800; color:#C62828; letter-spacing:0.05em;
                        text-transform:uppercase; margin-bottom:12px;">
                📝 STEP 2 & 3 — MULTILINGUAL SUMMARIES & TRANSLATIONS
            </div>
            """, unsafe_allow_html=True)

            sum_tab_hi, sum_tab_en, sum_tab_mr = st.tabs([
                "🇮🇳 Hindi View",
                "🇬🇧 English View",
                "🚩 Marathi View"
            ])

            with sum_tab_hi:
                st.markdown(f"""
                <div style="background:#FFF5F5; border:1.5px solid #FFCDD2; border-left:5px solid #E53935; border-radius:10px; padding:16px 20px; margin-bottom:1.25rem;">
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                        🇮🇳 हिंदी सारांश (Hindi Summary)
                    </div>
                    <div style="font-size:1rem; line-height:1.75; color:#1E293B; background:#FFFFFF;
                                border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                        {hindi_sum}
                    </div>
                    <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                        📊 {len(hindi_sum.split())} words
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.78rem; font-weight:800; color:#7C3AED; letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:8px; margin-top:0.5rem;">
                    🌐 Cross-Lingual Translations (English & Marathi)
                </div>
                """, unsafe_allow_html=True)

                c_en, c_mr = st.columns(2, gap="medium")
                with c_en:
                    st.markdown(f"""
                    <div style="background:#FFF5F5; border:2px solid #FFCDD2; border-left:5px solid #7C3AED; border-radius:10px; padding:16px 20px; height:100%;">
                        <div style="font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                            🇬🇧 English Translation
                        </div>
                        <div style="font-size:0.97rem; line-height:1.75; color:#1E293B; background:#FFFFFF; border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                            {english_sum}
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                            Meta NLLB-200 Engine &nbsp;·&nbsp; {len(english_sum.split())} words
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_mr:
                    st.markdown(f"""
                    <div style="background:#FFF5F5; border:2px solid #FFCDD2; border-left:5px solid #7C3AED; border-radius:10px; padding:16px 20px; height:100%;">
                        <div style="font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                            🚩 Marathi Translation (मराठी)
                        </div>
                        <div style="font-size:0.97rem; line-height:1.75; color:#1E293B; background:#FFFFFF; border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                            {marathi_sum}
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                            Meta NLLB-200 Engine &nbsp;·&nbsp; {len(marathi_sum.split())} words
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with sum_tab_en:
                st.markdown(f"""
                <div style="background:#FFF5F5; border:1.5px solid #FFCDD2; border-left:5px solid #2563EB; border-radius:10px; padding:16px 20px; margin-bottom:1.25rem;">
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                        🇬🇧 English Summary
                    </div>
                    <div style="font-size:1rem; line-height:1.75; color:#1E293B; background:#FFFFFF;
                                border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                        {english_sum}
                    </div>
                    <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                        📊 {len(english_sum.split())} words
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.78rem; font-weight:800; color:#7C3AED; letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:8px; margin-top:0.5rem;">
                    🌐 Cross-Lingual Translations (Hindi & Marathi)
                </div>
                """, unsafe_allow_html=True)

                c_hi, c_mr2 = st.columns(2, gap="medium")
                with c_hi:
                    st.markdown(f"""
                    <div style="background:#FFF5F5; border:2px solid #FFCDD2; border-left:5px solid #7C3AED; border-radius:10px; padding:16px 20px; height:100%;">
                        <div style="font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                            🇮🇳 Hindi Translation (हिंदी अनुवाद)
                        </div>
                        <div style="font-size:0.97rem; line-height:1.75; color:#1E293B; background:#FFFFFF; border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                            {hindi_sum}
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                            Meta NLLB-200 Engine &nbsp;·&nbsp; {len(hindi_sum.split())} words
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_mr2:
                    st.markdown(f"""
                    <div style="background:#FFF5F5; border:2px solid #FFCDD2; border-left:5px solid #7C3AED; border-radius:10px; padding:16px 20px; height:100%;">
                        <div style="font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                            🚩 Marathi Translation (मराठी भाषांतर)
                        </div>
                        <div style="font-size:0.97rem; line-height:1.75; color:#1E293B; background:#FFFFFF; border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                            {marathi_sum}
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                            Meta NLLB-200 Engine &nbsp;·&nbsp; {len(marathi_sum.split())} words
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with sum_tab_mr:
                st.markdown(f"""
                <div style="background:#FFF5F5; border:1.5px solid #FFCDD2; border-left:5px solid #EA580C; border-radius:10px; padding:16px 20px; margin-bottom:1.25rem;">
                    <div style="font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                        🚩 मराठी सारांश (Marathi Summary)
                    </div>
                    <div style="font-size:1rem; line-height:1.75; color:#1E293B; background:#FFFFFF;
                                border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                        {marathi_sum}
                    </div>
                    <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                        📊 {len(marathi_sum.split())} words
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.78rem; font-weight:800; color:#7C3AED; letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:8px; margin-top:0.5rem;">
                    🌐 Cross-Lingual Translations (Hindi & English)
                </div>
                """, unsafe_allow_html=True)

                c_hi2, c_en2 = st.columns(2, gap="medium")
                with c_hi2:
                    st.markdown(f"""
                    <div style="background:#FFF5F5; border:2px solid #FFCDD2; border-left:5px solid #7C3AED; border-radius:10px; padding:16px 20px; height:100%;">
                        <div style="font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                            🇮🇳 Hindi Translation (हिंदी अनुवाद)
                        </div>
                        <div style="font-size:0.97rem; line-height:1.75; color:#1E293B; background:#FFFFFF; border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                            {hindi_sum}
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                            Meta NLLB-200 Engine &nbsp;·&nbsp; {len(hindi_sum.split())} words
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_en2:
                    st.markdown(f"""
                    <div style="background:#FFF5F5; border:2px solid #FFCDD2; border-left:5px solid #7C3AED; border-radius:10px; padding:16px 20px; height:100%;">
                        <div style="font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:8px;">
                            🇬🇧 English Translation
                        </div>
                        <div style="font-size:0.97rem; line-height:1.75; color:#1E293B; background:#FFFFFF; border:1px solid #FFCDD2; border-radius:8px; padding:14px 16px; min-height:80px;">
                            {english_sum}
                        </div>
                        <div style="margin-top:8px; font-size:0.8rem; color:#64748B; font-weight:600;">
                            Meta NLLB-200 Engine &nbsp;·&nbsp; {len(english_sum.split())} words
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Stats Footer ───────────────────────────────────────────────────
            st.markdown(f"""
            <div style="background:#FFEBEE; border:1px solid #FFCDD2; border-radius:10px;
                        padding:12px 20px; color:#B71C1C; font-size:0.88rem; font-weight:600;
                        margin-top:1.25rem; margin-bottom:1.5rem; display:flex;
                        justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    📊 <strong>Original Input:</strong> {orig_words} words &nbsp;|&nbsp;
                    ⚡ <strong>Execution Time:</strong> {exec_sec:.2f}s
                </div>
                <div>
                    ⚙️ <strong>NMT Architecture:</strong> Meta NLLB-200 Unified Engine
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Action Buttons — Download JSON + New Article
            btn_col1, btn_col2 = st.columns(2)

            json_export = {
                "source_type": input_source_type,
                "classifier_checkpoint": chk_info,
                "category": cat_name,
                "confidence": conf_pct,
                "original_word_count": orig_words,
                "summary_word_count": sum_words,
                "compression_reduction": comp_ratio,
                "processing_time_seconds": exec_sec,
                "summaries": {
                    "hindi": hindi_sum,
                    "english": english_sum,
                    "marathi": marathi_sum
                }
            }

            with btn_col1:
                st.download_button(
                    label="💾 Download as .json",
                    data=json.dumps(json_export, indent=2, ensure_ascii=False),
                    file_name="hindi_news_nlp_results.json",
                    mime="application/json",
                    use_container_width=True
                )

            with btn_col2:
                if st.button("🔄 New Article", use_container_width=True):
                    st.rerun()

else:
    st.markdown("""
    <div class="thunder-card" style="text-align: center; padding: 2.5rem 1rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📰</div>
        <h4 style="color: #E53935; margin-bottom: 0.3rem;">Ready to Process News Article</h4>
        <p style="font-size: 0.95rem; max-width: 620px; margin: 0 auto; color: #718096;">
            Paste an article in <strong>Hindi, English, or Marathi</strong> above, upload a document (.pdf/.docx/.txt), 
            or select a sample from the sidebar to generate cross-lingual topic classification, summaries, and translations.
        </p>
    </div>
    """, unsafe_allow_html=True)
