import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import hf_hub_download
import PyPDF2
from io import BytesIO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Cambodia News Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 20px 30px;
        color: white;
        border-bottom: 3px solid #0f172a;
        margin-bottom: 30px;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 12px;
        color: #e0e7ff;
        margin: 5px 0 0 0;
        letter-spacing: 1px;
    }
    
    /* Input section */
    .input-section {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .input-label {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 15px;
        color: #1e293b;
    }
    
    .input-sublabel {
        font-size: 12px;
        color: #64748b;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    
    /* Results panel */
    .results-panel {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    
    .results-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1e293b;
    }
    
    .results-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 20px;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 40px 20px;
        color: #94a3b8;
    }
    
    .empty-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    
    /* Features list */
    .features-list {
        background: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        padding: 8px 0;
        color: #0f766e;
        font-size: 13px;
    }
    
    .feature-icon {
        margin-right: 10px;
        color: #14b8a6;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #bae6fd;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #0369a1;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #0c4a6e;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 0;
        color: #64748b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3b82f6;
        border-bottom: 3px solid #3b82f6;
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        padding: 15px;
        font-size: 14px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Spinner */
    .stSpinner {
        color: #3b82f6;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        color: #15803d;
    }
    
    .stError {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }
    
    .stInfo {
        background-color: #f0f9ff;
        border-left: 4px solid #0284c7;
        color: #082f49;
    }
    
    /* Chart styling */
    .stBar {
        border-radius: 8px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #94a3b8;
        font-size: 12px;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    """Load RoBERTa model from HuggingFace Hub"""
    # ⚠️ REPLACE 'your-username' with YOUR actual HuggingFace username
    model_path = hf_hub_download(
        repo_id="Theara2/cambodia-news-classifier",
        filename="roberta_best.pt"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        return None

def classify_text(text, model, tokenizer, device):
    """Classify input text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    
    pred_class = torch.argmax(probs, dim=1).item()
    pred_conf = probs[0, pred_class].item()
    
    return pred_class, pred_conf, probs

# ============================================================================
# LOAD MODEL AT START
# ============================================================================
model, tokenizer, device = load_model()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div class="header-container">
        <div class="header-title">🗞️ Cambodian News Classifier</div>
        <div class="header-subtitle">MULTI-CLASS AI ANALYSIS</div>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# NAVIGATION TABS
# ============================================================================
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 8])

with nav_col1:
    classifier_active = st.button("🤖 Classifier", use_container_width=True, key="nav_classifier")

with nav_col2:
    history_active = st.button("📊 Session History", use_container_width=True, key="nav_history")

with nav_col3:
    about_active = st.button("ℹ️ About", use_container_width=True, key="nav_about")

st.markdown("---")

# ============================================================================
# MAIN CLASSIFIER PAGE
# ============================================================================
col_left, col_right = st.columns([1, 1], gap="large")

# ============================================================================
# LEFT COLUMN - INPUT SECTION
# ============================================================================
with col_left:
    st.markdown("""
        <div class="input-section">
            <div class="input-label">📄 Input Section</div>
            <div class="input-sublabel">Paste news text for classification</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs for input method
    tab1, tab2 = st.tabs(["📝 Text Input", "📤 PDF Upload"])
    
    with tab1:
        text_input = st.text_area(
            "Direct Text Entry",
            height=250,
            placeholder="Paste your news article here ...\n\nThe government announced new economic policies today ...",
            label_visibility="collapsed"
        )
        input_text = text_input
    
    with tab2:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text:
                st.success(f"✅ Extracted {len(extracted_text)} characters from PDF")
                input_text = extracted_text
            else:
                st.error("❌ Could not extract text from PDF")
                input_text = ""
        else:
            input_text = ""
    
    # Analyze button
    if st.button("🔍 Analyze Text", use_container_width=True, key="analyze_btn"):
        if input_text.strip():
            with st.spinner("⏳ Analyzing article..."):
                pred_class, pred_conf, probs = classify_text(
                    input_text,
                    model,
                    tokenizer,
                    device
                )
            
            # Store results in session state
            st.session_state.last_prediction = {
                'class': pred_class,
                'confidence': pred_conf,
                'probs': probs,
                'text': input_text,
                'text_length': len(input_text),
                'word_count': len(input_text.split())
            }
        else:
            st.error("❌ Please enter a news article to analyze")

# ============================================================================
# RIGHT COLUMN - RESULTS PANEL
# ============================================================================
with col_right:
    st.markdown("""
        <div class="results-panel">
            <div class="results-title">📊 Results Panel</div>
    """, unsafe_allow_html=True)
    
    if 'last_prediction' in st.session_state:
        prediction = st.session_state.last_prediction
        
        st.markdown("""
            <div class="results-subtitle">
                Enter a news article on the left and click "Analyze Text" to see classification results, confidence scores, and detailed analytics.
            </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📌 PREDICTED CATEGORY</div>
                    <div class="metric-value">Class {prediction['class']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 CONFIDENCE SCORE</div>
                    <div class="metric-value">{prediction['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Category probabilities
        st.subheader("Category Probabilities")
        prob_data = {
            f"Class {i}": float(p) 
            for i, p in enumerate(prediction['probs'][0].tolist())
        }
        st.bar_chart(prob_data)
        
        # Text statistics
        st.subheader("📊 Text Statistics")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Characters", prediction['text_length'])
        with stat_col2:
            st.metric("Words", prediction['word_count'])
        
        # Success message
        st.success("✅ Classification Complete!")
        
    else:
        st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">📈</div>
                <div style="font-weight: 600; margin-bottom: 10px;">Results Panel</div>
                <div>Enter a news article on the left and click "Analyze Text" to see classification results, confidence scores, and detailed analytics.</div>
                <div class="features-list">
                    <div class="feature-item"><span class="feature-icon">✓</span> Supports news text input</div>
                    <div class="feature-item"><span class="feature-icon">✓</span> 6-category classification model</div>
                    <div class="feature-item"><span class="feature-icon">✓</span> Confidence scores for all categories</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div class="footer">
        <p>Made with ❤️ | Cambodia News Classification System</p>
        <p>Powered by RoBERTa + Streamlit</p>
    </div>
""", unsafe_allow_html=True)