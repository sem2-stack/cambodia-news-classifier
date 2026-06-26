# ============================================================================
# DEBUG VERSION - Find Correct Category Mapping
# ============================================================================

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
# TEMPORARY: Use generic names to find the correct mapping
# ============================================================================
CATEGORY_NAMES = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
    5: "Class 5"
}

CATEGORY_COLORS = {
    "Class 0": "#3b82f6",
    "Class 1": "#8b5cf6",
    "Class 2": "#10b981",
    "Class 3": "#ef4444",
    "Class 4": "#f59e0b",
    "Class 5": "#14b8a6"
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
    .debug-box {
        background: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
    }
    .debug-class {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        margin: 2px;
        font-family: monospace;
        font-size: 14px;
    }
    .debug-highlight {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
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
    
    # Return all probabilities for debugging
    all_probs = {i: p.item() for i, p in enumerate(probs[0])}
    
    return pred_class, pred_name, pred_conf, probs, all_probs

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
        <div class="header-title">🔍 Category Mapping Debug Tool</div>
        <div class="header-subtitle">FIND THE CORRECT CATEGORY ORDER</div>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# DEBUG INFO
# ============================================================================
st.info("""
**📋 How to use this debug tool:**

1. Paste a **Politics** article in the text area below
2. Click "Analyze Text" 
3. Look at the **Class Probabilities** section
4. Note which Class (0-5) gets the highest probability
5. Update the `CATEGORY_NAMES` dictionary with the correct mapping
6. Test with articles from other categories (Technology, Economics, Health, Sports, Environment)
""")

# ============================================================================
# MAIN CLASSIFIER PAGE
# ============================================================================
col_left, col_right = st.columns([1, 1], gap="large")

# LEFT COLUMN - INPUT SECTION
with col_left:
    st.markdown("""
        <div class="input-section">
            <div class="input-label">📄 Input Section</div>
            <div class="input-sublabel">Paste news text for classification</div>
        </div>
    """, unsafe_allow_html=True)
    
    text_input = st.text_area(
        "Direct Text Entry",
        height=300,
        placeholder="Paste your news article here ...",
        label_visibility="collapsed",
        key="text_input_area"
    )
    
    if st.button("🔍 Analyze Text", use_container_width=True, key="analyze_btn"):
        text_input = st.session_state.get("text_input_area", "")
        
        if not text_input or not text_input.strip():
            st.error("❌ Please enter a news article to analyze")
        else:
            with st.spinner("⏳ Analyzing article..."):
                pred_class, pred_name, pred_conf, probs, all_probs = classify_text(
                    text_input,
                    model,
                    tokenizer,
                    device
                )
            
            st.session_state.last_prediction = {
                'class': pred_class,
                'category_name': pred_name,
                'confidence': pred_conf,
                'probs': probs,
                'all_probs': all_probs,
                'text': text_input,
                'text_length': len(text_input),
                'word_count': len(text_input.split()),
                'timestamp': datetime.now().strftime("%I:%M %p")
            }
            
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
        """, unsafe_allow_html=True)
        
        # ============================================================
        # DEBUG: Show all class probabilities
        # ============================================================
        st.markdown("""
            <div class="debug-box">
                <h4>🔍 Debug: Class Probabilities</h4>
                <p style="font-size: 13px; color: #64748b;">
                    The highest probability shows which class the model thinks this article belongs to.
                    <br><strong>Note which Class (0-5) gets the highest score for each category.</strong>
                </p>
        """, unsafe_allow_html=True)
        
        # Create a DataFrame for better display
        debug_data = []
        for class_idx, prob in prediction['all_probs'].items():
            is_top = class_idx == prediction['class']
            debug_data.append({
                'Class': f"Class {class_idx}",
                'Probability': f"{prob:.1%}",
                'Bar': prob,
                'Top': '⭐' if is_top else ''
            })
        
        # Sort by probability descending
        debug_data.sort(key=lambda x: x['Bar'], reverse=True)
        
        for item in debug_data:
            bar_color = '#f59e0b' if item['Top'] else '#94a3b8'
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                    <span style="min-width: 80px; font-weight: {'bold' if item['Top'] else 'normal'};">
                        {item['Class']} {item['Top']}
                    </span>
                    <span style="min-width: 60px;">{item['Probability']}</span>
                    <div style="flex: 1;">
                        <div class="confidence-bar">
                            <div class="confidence-bar-fill" style="width: {item['Bar']*100}%; background: {bar_color};"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
            <div style="margin-top: 10px; padding: 10px; background: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <strong>💡 Next Step:</strong>
                <br>If this was a <strong>Politics</strong> article and <strong>Class X</strong> got the highest score,
                then update your <code>CATEGORY_NAMES</code> with: <br>
                <code style="display: block; margin-top: 5px; background: #1e293b; color: #e2e8f0; padding: 8px; border-radius: 4px;">
                CATEGORY_NAMES = {{<br>
                    X: "Politics",  # ← X is the class number you see above<br>
                    ...<br>
                }}
                </code>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence Scores for all categories
        st.subheader("All Category Probabilities")
        
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
        
    else:
        st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">📈</div>
                <div style="font-weight: 600; margin-bottom: 10px;">Results Panel</div>
                <div>Enter a news article and click "Analyze Text" to see the class probabilities.</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# INSTRUCTIONS
# ============================================================================
st.markdown("""
    <div style="margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
        <h4>📋 How to Find the Correct Mapping</h4>
        <ol style="line-height: 2;">
            <li>Paste a <strong>Politics</strong> article and click "Analyze Text"</li>
            <li>Look at the <strong>Debug: Class Probabilities</strong> section</li>
            <li>Note which <strong>Class (0-5)</strong> gets the highest score</li>
            <li>That class number = Politics in your model</li>
            <li>Repeat for <strong>Technology, Economics, Health, Sports, Environment</strong></li>
            <li>Update <code>CATEGORY_NAMES</code> with the correct mapping</li>
        </ol>
        <div style="margin-top: 10px; padding: 10px; background: #dbeafe; border-radius: 8px;">
            <strong>Example mapping (after testing):</strong><br>
            <code>
            CATEGORY_NAMES = {<br>
                0: "Politics",  # ← Highest score for Politics articles<br>
                1: "Technology",  # ← Highest score for Technology articles<br>
                2: "Economics",  # ← Highest score for Economics articles<br>
                3: "Health",  # ← Highest score for Health articles<br>
                4: "Sports",  # ← Highest score for Sports articles<br>
                5: "Environment"  # ← Highest score for Environment articles<br>
            }
            </code>
        </div>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div class="footer">
        <p>🔍 Debug Tool | Cambodia News Classification System</p>
    </div>
""", unsafe_allow_html=True)
