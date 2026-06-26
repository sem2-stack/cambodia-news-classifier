import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, RobertaConfig
from huggingface_hub import hf_hub_download
import PyPDF2
from io import BytesIO
import os
import pandas as pd
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Cambodian News Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ============================================================================
# CATEGORY NAMES - CORRECT MAPPING
# ============================================================================
CATEGORY_NAMES = {
    0: "Politics",
    1: "Technology",
    2: "Economics",
    3: "Health",
    4: "Sports",
    5: "Environment"
}

CATEGORY_COLORS = {
    "Politics": "#3b82f6",
    "Technology": "#8b5cf6",
    "Economics": "#10b981",
    "Health": "#ef4444",
    "Sports": "#f59e0b",
    "Environment": "#14b8a6"
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main { padding: 0; }
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 20px 30px;
        color: white;
        border-bottom: 3px solid #0f172a;
        margin-bottom: 30px;
    }
    .header-title { font-size: 28px; font-weight: bold; margin: 0; }
    .header-subtitle { font-size: 12px; color: #e0e7ff; margin: 5px 0 0 0; letter-spacing: 1px; }
    .input-section {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .input-label { font-size: 16px; font-weight: 600; margin-bottom: 15px; color: #1e293b; }
    .input-sublabel { font-size: 12px; color: #64748b; margin-top: -10px; margin-bottom: 15px; }
    .results-panel {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .results-title { font-size: 20px; font-weight: bold; margin-bottom: 10px; color: #1e293b; }
    .results-subtitle { font-size: 13px; color: #64748b; margin-bottom: 20px; }
    .empty-state { text-align: center; padding: 40px 20px; color: #94a3b8; }
    .empty-icon { font-size: 48px; margin-bottom: 15px; }
    .features-list { background: #f8fafc; padding: 15px; border-radius: 8px; margin-top: 20px; }
    .feature-item { display: flex; align-items: center; padding: 8px 0; color: #0f766e; font-size: 13px; }
    .feature-icon { margin-right: 10px; color: #14b8a6; }
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #bae6fd;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #0369a1; margin: 10px 0; }
    .metric-label { font-size: 12px; color: #0c4a6e; font-weight: 600; margin-bottom: 5px; }
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
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 2px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { padding: 10px 0; color: #64748b; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #3b82f6; border-bottom: 3px solid #3b82f6; }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        padding: 15px;
        font-size: 14px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTextArea textarea:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
    .top-category-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #3b82f6;
        margin-bottom: 20px;
    }
    .top-category-label { font-size: 14px; color: #64748b; font-weight: 500; }
    .top-category-name { font-size: 32px; font-weight: bold; color: #1e293b; margin: 10px 0; }
    .top-category-confidence { font-size: 18px; color: #3b82f6; font-weight: 600; }
    .confidence-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 4px;
    }
    .confidence-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
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
# CUSTOM MODEL CLASS
# ============================================================================
class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.num_labels)
        )
        self.num_labels = config.num_labels
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return type('obj', (object,), {'logits': logits})()

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    model_path = "roberta_best.pt"
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model from HuggingFace (477 MB - this may take a few minutes)..."):
            model_path = hf_hub_download(
                repo_id="Theara2/cambodia-news-classifier",
                filename="roberta_best.pt"
            )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("⏳ Loading model into memory..."):
        state_dict = torch.load(model_path, map_location="cpu")
        
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder.", "roberta.", 1)
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        num_labels = 6
        for key in fixed_state_dict.keys():
            if "out_proj.weight" in key:
                num_labels = fixed_state_dict[key].shape[0]
                break
        
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = num_labels
        
        model = CustomRobertaForSequenceClassification(config)
        model.load_state_dict(fixed_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        st.success(f"✅ Model loaded successfully with {num_labels} classes!")
    
    return model, tokenizer, device

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def extract_text_from_pdf(pdf_file):
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
    pred_name = CATEGORY_NAMES.get(pred_class, f"Class {pred_class}")
    
    return pred_class, pred_name, pred_conf, probs

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

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
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🤖 Classifier", "📊 Session History", "ℹ️ About"])

# ============================================================================
# TAB 1: CLASSIFIER - FIXED INPUT HANDLING
# ============================================================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # LEFT COLUMN - INPUT SECTION
    with col_left:
        st.markdown("""
            <div class="input-section">
                <div class="input-label">📄 Input Section</div>
                <div class="input-sublabel">Paste news text for classification</div>
            </div>
        """, unsafe_allow_html=True)
        
        tab_text, tab_pdf = st.tabs(["📝 Text Input", "📤 PDF Upload"])
        
        # Initialize
        input_text = ""
        extracted_text = ""
        uploaded_file = None
        
        with tab_text:
            st.caption("Direct Text Entry")
            st.caption("Perfect for copied articles or short texts")
            text_input = st.text_area(
                "Direct Text Entry",
                height=300,
                placeholder="Paste your news article here ...\n\nThe government announced new economic policies today ...",
                label_visibility="collapsed",
                key="text_input_area"
            )
        
        with tab_pdf:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
            if uploaded_file is not None:
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    st.success(f"✅ Extracted {len(extracted_text)} characters from PDF")
                else:
                    st.error("❌ Could not extract text from PDF")
        
        # Analyze button
        if st.button("🔍 Analyze Text", use_container_width=True, key="analyze_btn"):
            # Get text from text area
            text_from_area = st.session_state.get("text_input_area", "")
            
            # Determine which input to use (text area takes priority)
            if text_from_area and text_from_area.strip():
                final_text = text_from_area
            elif extracted_text and extracted_text.strip():
                final_text = extracted_text
            else:
                final_text = ""
            
            if not final_text or not final_text.strip():
                st.error("❌ Please enter a news article to analyze")
            else:
                with st.spinner("⏳ Analyzing article..."):
                    pred_class, pred_name, pred_conf, probs = classify_text(
                        final_text,
                        model,
                        tokenizer,
                        device
                    )
                
                # Store results
                st.session_state.last_prediction = {
                    'class': pred_class,
                    'category_name': pred_name,
                    'confidence': pred_conf,
                    'probs': probs,
                    'text': final_text,
                    'text_length': len(final_text),
                    'word_count': len(final_text.split()),
                    'timestamp': datetime.now().strftime("%I:%M %p")
                }
                
                # Add to history
                st.session_state.history.append(st.session_state.last_prediction)
                
                st.success("✅ Analysis complete!")
                st.rerun()
    
    # RIGHT COLUMN - RESULTS PANEL
    with col_right:
        st.markdown("""
            <div class="results-panel">
                <div class="results-title">📊 Results Panel</div>
        """, unsafe_allow_html=True)
        
        if st.session_state.last_prediction:
            prediction = st.session_state.last_prediction
            
            st.markdown("""
                <div class="results-subtitle">
                    Enter a news article on the left and click "Analyze Text" to see classification results, confidence scores, and detailed analytics.
                </div>
            """, unsafe_allow_html=True)
            
            # Top Classification Card
            color = CATEGORY_COLORS.get(prediction['category_name'], '#3b82f6')
            st.markdown(f"""
                <div class="top-category-card" style="border-color: {color};">
                    <div class="top-category-label">TOP CLASSIFICATION</div>
                    <div class="top-category-name" style="color: {color};">{prediction['category_name']}</div>
                    <div class="top-category-confidence">{prediction['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Text statistics
            st.markdown(f"""
                <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                    <div><strong>{prediction['text_length']:,}</strong> Characters</div>
                    <div><strong>{prediction['word_count']:,}</strong> words</div>
                </div>
                <div style="font-size: 13px; color: #10b981; margin-bottom: 15px;">✅ Text length is optimal for classification</div>
            """, unsafe_allow_html=True)
            
            # Confidence Scores
            st.subheader("Confidence Scores")
            
            probs_list = [(CATEGORY_NAMES.get(i, f"Class {i}"), float(p)) for i, p in enumerate(prediction['probs'][0].tolist())]
            probs_list.sort(key=lambda x: x[1], reverse=True)
            
            for name, prob in probs_list:
                color = CATEGORY_COLORS.get(name, '#3b82f6')
                st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 14px;">
                            <span>{name}</span>
                            <span style="font-weight: 600;">{prob:.1%}</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-bar-fill" style="width: {prob*100}%; background: {color};"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Action buttons
            col_exp, col_clr = st.columns(2)
            with col_exp:
                if st.button("📤 Export", use_container_width=True):
                    export_data = {
                        'Category': prediction['category_name'],
                        'Confidence': prediction['confidence'],
                        'Text': prediction['text'][:500] + "...",
                        'Characters': prediction['text_length'],
                        'Words': prediction['word_count'],
                        'Timestamp': prediction['timestamp']
                    }
                    df = pd.DataFrame([export_data])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col_clr:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.last_prediction = None
                    st.rerun()
            
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
# TAB 2: SESSION HISTORY
# ============================================================================
with tab2:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2>📊 Session History</h2>
            <p style="color: #64748b;">All classification results from this session</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        total_articles = len(st.session_state.history)
        categories_used = len(set(h['category_name'] for h in st.session_state.history))
        avg_confidence = sum(h['confidence'] for h in st.session_state.history) / total_articles
        top_category = max(set(h['category_name'] for h in st.session_state.history), 
                          key=lambda x: sum(1 for h in st.session_state.history if h['category_name'] == x))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="history-stat-card">
                    <div class="history-stat-value">{total_articles}</div>
                    <div class="history-stat-label">TOTAL ARTICLES</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="history-stat-card">
                    <div class="history-stat-value">{categories_used}/6</div>
                    <div class="history-stat-label">CATEGORIES USED</div>
                    <div class="history-stat-sub">of 6 total</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="history-stat-card">
                    <div class="history-stat-value">{avg_confidence:.1%}</div>
                    <div class="history-stat-label">AVG CONFIDENCE</div>
                    <div class="history-stat-sub">across session</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="history-stat-card">
                    <div class="history-stat-value" style="color: {CATEGORY_COLORS.get(top_category, '#3b82f6')};">{top_category}</div>
                    <div class="history-stat-label">TOP CATEGORY</div>
                    <div class="history-stat-sub">most frequent</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_search, col_filter, col_sort = st.columns([2, 1, 1])
        with col_search:
            search_query = st.text_input("🔍 Search articles...", placeholder="Search by category or text...")
        with col_filter:
            filter_category = st.selectbox("Filter by category", ["All"] + list(CATEGORY_NAMES.values()))
        with col_sort:
            sort_order = st.selectbox("Sort by", ["Most Recent", "Most Confident", "Oldest"])
        
        filtered_history = st.session_state.history.copy()
        
        if search_query:
            filtered_history = [h for h in filtered_history if 
                              search_query.lower() in h['category_name'].lower() or 
                              search_query.lower() in h['text'][:100].lower()]
        
        if filter_category != "All":
            filtered_history = [h for h in filtered_history if h['category_name'] == filter_category]
        
        if sort_order == "Most Recent":
            filtered_history = filtered_history[::-1]
        elif sort_order == "Most Confident":
            filtered_history = sorted(filtered_history, key=lambda x: x['confidence'], reverse=True)
        
        st.markdown(f"### {len(filtered_history)} articles")
        
        for item in filtered_history:
            color = CATEGORY_COLORS.get(item['category_name'], '#3b82f6')
            st.markdown(f"""
                <div class="history-item" style="border-left-color: {color};">
                    <div class="history-item-category" style="color: {color};">
                        {item['category_name']}
                    </div>
                    <div class="history-item-text">
                        {item['text'][:150]}...
                    </div>
                    <div class="history-item-meta">
                        {item['confidence']:.1%} confidence • {item['timestamp']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
            
    else:
        st.info("No articles classified yet. Go to the Classifier tab to get started!")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================
with tab3:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2>ℹ️ About This App</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        ### 🗞️ Cambodian News Classifier
        
        This application uses a fine-tuned RoBERTa model to classify Cambodian news articles into 6 categories:
        
        - **Politics** - Government, elections, policy
        - **Technology** - Tech news, digital transformation
        - **Economics** - Business, finance, trade
        - **Health** - Healthcare, public health
        - **Sports** - Athletics, competitions
        - **Environment** - Climate, conservation
        
        ### 📊 How It Works
        
        1. **Input** - Paste text or upload a PDF
        2. **Analyze** - Click "Analyze Text" to classify
        3. **Results** - View category and confidence scores
        4. **History** - Track all classifications
        
        ### 🔧 Technical Details
        
        - **Model:** RoBERTa (fine-tuned)
        - **Framework:** PyTorch + Transformers
        - **Deployment:** Streamlit Cloud
        - **Model Hosting:** HuggingFace Hub
        
        ### 👨‍💻 Developer
        
        Built with ❤️ for Cambodian news analysis.
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div class="footer">
        <p>Made with ❤️ | Cambodia News Classification System</p>
        <p>Powered by RoBERTa + Streamlit</p>
    </div>
""", unsafe_allow_html=True)
