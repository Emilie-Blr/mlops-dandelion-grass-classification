"""
🌟 Streamlit Web Application for Plant Classification - PRO VERSION 🌟
Features: Dark Mode, Webcam, Batch Processing, History, Advanced Visualizations
"""

import io
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import base64

from PIL import Image, ImageEnhance
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


# ==================== CONFIGURATION ====================
API_URL = "http://localhost:8000"
MODEL_PATH = Path("models/best_model.pth")
CLASS_NAMES = ["Dandelion", "Grass"]
CLASS_EMOJIS = {"Dandelion": "🌼", "Grass": "🌿"}
CLASS_COLORS = {"Dandelion": "#FFD700", "Grass": "#90EE90"}
HISTORY_FILE = Path("data/prediction_history.json")


# ==================== MODEL FUNCTIONS ====================
class DandelionGrassClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # On recrée exactement la même structure que lors de l'entraînement
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        
        # La "tête" du modèle doit correspondre à celle définie dans training/model.py
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(num_classes=2):
    """Create the custom model structure"""
    return DandelionGrassClassifier(num_classes=num_classes)


@st.cache_resource
def load_model_from_checkpoint(model_path: str, device: torch.device):
    """Load model from checkpoint with caching"""
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def get_image_transforms():
    """Get preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_with_api(image_bytes, api_url):
    """Make prediction using API"""
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    response = requests.post(f"{api_url}/predict", files=files, timeout=30)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")


def predict_local(image, model, device):
    """Make prediction using local model"""
    transform = get_image_transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        "predicted_class": CLASS_NAMES[predicted.item()].lower(),
        "confidence": confidence.item(),
        "probabilities": probabilities[0].cpu().numpy()
    }


# ==================== VISUALIZATION FUNCTIONS ====================
def create_confidence_chart(probs, class_names, dark_mode=False):
    """Create animated confidence bar chart"""
    colors = [CLASS_COLORS[name] for name in class_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs * 100,
            y=class_names,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white' if dark_mode else 'black', width=2),
                pattern_shape="",
            ),
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
            textfont=dict(size=16, color='black', family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Confidence Score",
            font=dict(size=20, color='white' if dark_mode else 'black', family='Arial Black')
        ),
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        xaxis=dict(
            range=[0, 100],
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            tickfont=dict(size=16, family='Arial', color='white' if dark_mode else 'black')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if dark_mode else 'black')
    )
    
    return fig


def create_gauge_chart(confidence, predicted_class, dark_mode=False):
    """Create gauge chart for confidence"""
    color = CLASS_COLORS[predicted_class.capitalize()]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence Level", 'font': {'size': 20, 'color': 'white' if dark_mode else 'black'}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': color},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,0,0,0.1)'},
                {'range': [50, 75], 'color': 'rgba(255,255,0,0.1)'},
                {'range': [75, 100], 'color': 'rgba(0,255,0,0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white' if dark_mode else 'black', 'family': "Arial"}
    )
    
    return fig


def create_history_chart(history_df, dark_mode=False):
    """Create timeline chart from prediction history"""
    if history_df.empty:
        return None
    
    fig = px.scatter(
        history_df, 
        x='timestamp', 
        y='confidence',
        color='predicted_class',
        size='confidence',
        hover_data=['confidence'],
        title='Prediction History',
        color_discrete_map={'dandelion': CLASS_COLORS['Dandelion'], 'grass': CLASS_COLORS['Grass']}
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if dark_mode else 'black'),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', range=[0, 1])
    )
    
    return fig


# ==================== HISTORY FUNCTIONS ====================
def load_history() -> List[Dict]:
    """Load prediction history from JSON"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_prediction(predicted_class: str, confidence: float, prediction_time: float):
    """Save prediction to history"""
    history = load_history()
    
    prediction = {
        'timestamp': datetime.now().isoformat(),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'prediction_time': prediction_time
    }
    
    history.append(prediction)
    
    # Keep only last 100 predictions
    history = history[-100:]
    
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def get_history_stats() -> Dict:
    """Calculate statistics from history"""
    history = load_history()
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    return {
        'total_predictions': len(history),
        'avg_confidence': df['confidence'].mean(),
        'avg_time': df['prediction_time'].mean(),
        'dandelion_count': len(df[df['predicted_class'] == 'dandelion']),
        'grass_count': len(df[df['predicted_class'] == 'grass']),
    }


# ==================== UI HELPER FUNCTIONS ====================
def get_base64_image(image_path):
    """Convert image to base64 for CSS background"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def apply_custom_css(dark_mode=False):
    """Apply custom CSS styling"""
    bg_color = "#0E1117" if dark_mode else "#FFFFFF"
    text_color = "#FAFAFA" if dark_mode else "#262730"
    card_bg = "#1E1E1E" if dark_mode else "#F0F2F6"
    
    st.markdown(f"""
    <style>
        /* Global Styles */
        .main {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        /* Animated Title */
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
        }}
        
        .big-title {{
            font-size: 48px !important;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(45deg, #FFD700, #90EE90, #87CEEB);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: float 3s ease-in-out infinite;
            margin: 20px 0;
        }}
        
        /* Result Cards */
        .result-card {{
            padding: 30px;
            border-radius: 20px;
            margin: 20px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideIn 0.5s ease-out;
        }}
        
        .result-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.3);
        }}
        
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .dandelion-card {{
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            border: 3px solid #FF8C00;
        }}
        
        .grass-card {{
            background: linear-gradient(135deg, #90EE90 0%, #32CD32 100%);
            border: 3px solid #228B22;
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: {card_bg};
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }}
        
        /* Buttons */
        .stButton>button {{
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #FFD700 0%, #90EE90 100%);
        }}
        
        /* File Uploader */
        .uploadedFile {{
            border-radius: 10px;
            border: 2px dashed #90EE90;
        }}
        
        /* Sidebar */
        .css-1d391kg {{
            background-color: {card_bg};
        }}
        
        /* Badge */
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            margin: 5px;
        }}
        
        .badge-success {{
            background-color: #28a745;
            color: white;
        }}
        
        .badge-warning {{
            background-color: #ffc107;
            color: black;
        }}
        
        .badge-info {{
            background-color: #17a2b8;
            color: white;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .stat-box {{
            background: {card_bg};
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #FFD700;
        }}
        
        .stat-label {{
            font-size: 14px;
            color: {text_color};
            margin-top: 5px;
        }}
    </style>
    """, unsafe_allow_html=True)


# ==================== MAIN APP ====================
def main():
    # Page config
    st.set_page_config(
        page_title="🌟 Plant Classifier PRO",
        page_icon="🌼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = None
    
    # Apply CSS
    apply_custom_css(st.session_state.dark_mode)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("# ⚙️ Settings")
        
        # Dark Mode Toggle
        st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        
        st.markdown("---")
        
        # Mode Selection
        app_mode = st.radio(
            "Select Mode",
            ["Single Image", "Webcam (Coming Soon)", "Batch Processing", "History & Stats"],
            index=0
        )
        
        st.markdown("---")
        
        # API Settings
        use_api = st.checkbox("Use API", value=False, help="Use FastAPI backend for predictions")
        
        if use_api:
            api_url = st.text_input("API URL", value=API_URL)
            
            if st.button("Test Connection", use_container_width=True):
                with st.spinner("Testing..."):
                    try:
                        response = requests.get(f"{api_url}/health", timeout=5)
                        if response.status_code == 200:
                            st.success("✅ API Connected!")
                        else:
                            st.error(f"❌ Status {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Connection failed")
        else:
            st.info("Local Model Mode")
            if MODEL_PATH.exists():
                model_size = MODEL_PATH.stat().st_size / (1024 * 1024)
                st.success(f"Model loaded ({model_size:.1f} MB)")
            else:
                st.warning(f"Model not found")
        
        st.markdown("---")
        
        # Model Info
        st.markdown("### Model Info")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class='badge badge-info'>ResNet18</div>
            <div class='badge badge-success'>PyTorch</div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class='badge badge-warning'>224x224</div>
            <div class='badge badge-info'>2 Classes</div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Stats
        stats = get_history_stats()
        if stats:
            st.markdown("### Quick Stats")
            st.metric("Total Predictions", stats['total_predictions'])
            st.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")
            st.metric("Avg Time", f"{stats['avg_time']:.3f}s")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 11px;'>
            <p>MLOps Project PRO 2025</p>
            <p>Made with using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== HEADER ====================
    st.markdown("<h1 class='big-title'>🌼 Plant Classifier PRO 🌿</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>Advanced AI-powered plant classification with real-time analysis</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ==================== MAIN CONTENT ====================
    if app_mode == "Single Image":
        render_single_image_mode(use_api, api_url if use_api else None)
    
    elif app_mode == "Webcam (Coming Soon)":
        render_webcam_mode()
    
    elif app_mode == "Batch Processing":
        render_batch_mode(use_api, api_url if use_api else None)
    
    elif app_mode == "History & Stats":
        render_history_mode()


# ==================== MODE RENDERERS ====================
def render_single_image_mode(use_api, api_url):
    """Render single image prediction mode"""
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("### Upload Image")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of dandelion or grass"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image enhancement options
            with st.expander("Image Adjustments (Optional)"):
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                
                if brightness != 1.0 or contrast != 1.0:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(brightness)
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(contrast)
                    st.image(image, caption="Enhanced Image", use_container_width=True)
            
            # Image info
            file_size = uploaded_file.size / 1024
            st.markdown(f"""
            <div class='metric-card'>
                <strong>Image Info</strong><br>
                Size: {image.size[0]}x{image.size[1]}<br>
                File: {file_size:.1f} KB<br>
                Format: {image.format}<br>
                Mode: {image.mode}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Prediction Result")
        
        if uploaded_file is not None:
            predict_button = st.button("Classify Now!", type="primary", use_container_width=True)
            
            if predict_button:
                with st.spinner("Analyzing image..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    start_time = time.time()
                    
                    try:
                        if use_api:
                            # API prediction
                            image_bytes = io.BytesIO()
                            image.save(image_bytes, format='JPEG')
                            image_bytes.seek(0)
                            
                            result = predict_with_api(image_bytes, api_url)
                            predicted_class = result["predicted_class"]
                            confidence = result["confidence"]
                            probs = np.array([1-confidence, confidence]) if predicted_class == "grass" else np.array([confidence, 1-confidence])
                        else:
                            # Local prediction
                            if not MODEL_PATH.exists():
                                st.error("❌ Model not found! Please train the model first.")
                                st.stop()
                            
                            device = torch.device("mps" if torch.backends.mps.is_available() else 
                                                "cuda" if torch.cuda.is_available() else "cpu")
                            
                            if st.session_state.model_loaded is None:
                                with st.spinner("Loading model..."):
                                    st.session_state.model_loaded = load_model_from_checkpoint(str(MODEL_PATH), device)
                            
                            result = predict_local(image, st.session_state.model_loaded, device)
                            predicted_class = result["predicted_class"]
                            confidence = result["confidence"]
                            probs = result["probabilities"]
                        
                        prediction_time = time.time() - start_time
                        
                        # Save to history
                        save_prediction(predicted_class, confidence, prediction_time)
                        
                        # Display result
                        pred_class_title = predicted_class.capitalize()
                        emoji = CLASS_EMOJIS.get(pred_class_title, "🌱")
                        
                        st.markdown(f"<p style='font-size: 48px; text-align: center; margin: 0;'>{emoji}</p>", unsafe_allow_html=True)
                        
                        # Result card
                        card_class = "dandelion-card" if predicted_class == "dandelion" else "grass-card"
                        st.markdown(f"""
                        <div class='result-card {card_class}'>
                            <h1 style='text-align: center; margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
                                {pred_class_title}
                            </h1>
                            <hr style='margin: 15px 0; border: 2px solid white;'>
                            <p style='text-align: center; font-size: 36px; margin: 10px 0; color: white;'>
                                <strong>{confidence * 100:.2f}%</strong>
                            </p>
                            <p style='text-align: center; font-size: 18px; color: rgba(255,255,255,0.9); margin: 5px 0;'>
                                ⚡ {prediction_time:.3f}s
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        tab1, tab2 = st.tabs(["Bar Chart", "Gauge"])
                        
                        with tab1:
                            st.plotly_chart(
                                create_confidence_chart(probs, CLASS_NAMES, st.session_state.dark_mode),
                                use_container_width=True
                            )
                        
                        with tab2:
                            st.plotly_chart(
                                create_gauge_chart(confidence, predicted_class, st.session_state.dark_mode),
                                use_container_width=True
                            )
                        
                        # Detailed info
                        with st.expander("Detailed Analysis"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.metric("🌼 Dandelion", f"{probs[0]*100:.2f}%", 
                                         delta=f"{(probs[0]-0.5)*100:.1f}%" if probs[0] > 0.5 else None)
                                st.metric("Inference Time", f"{prediction_time*1000:.1f} ms")
                            
                            with col_b:
                                st.metric("🌿 Grass", f"{probs[1]*100:.2f}%",
                                         delta=f"{(probs[1]-0.5)*100:.1f}%" if probs[1] > 0.5 else None)
                                device_name = 'GPU (MPS)' if torch.backends.mps.is_available() else 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
                                st.metric("Device", device_name)
                        
                        st.success("✅ Classification complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("Error Details"):
                            st.exception(e)
        else:
            st.info("Upload an image to get started")
            
            # Example section
            st.markdown("### What to Upload?")
            
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                st.markdown("""
                <div class='metric-card'>
                    <h3>🌼 Dandelion</h3>
                    <ul style='text-align: left;'>
                        <li>Bright yellow flowers</li>
                        <li>White fluffy seed heads</li>
                        <li>Single flower per stem</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with ex_col2:
                st.markdown("""
                <div class='metric-card'>
                    <h3>🌿 Grass</h3>
                    <ul style='text-align: left;'>
                        <li>Green blades</li>
                        <li>Lawn or field views</li>
                        <li>Ground cover</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)


def render_webcam_mode():
    """Render webcam mode (placeholder)"""
    st.markdown("### Webcam Mode")
    
    st.info("🚧 Coming Soon! This feature will allow real-time classification using your webcam.")
    
    st.markdown("""
    <div class='metric-card'>
        <h3>Planned Features:</h3>
        <ul style='text-align: left;'>
            <li>Real-time video feed</li>
            <li>Live predictions</li>
            <li>Confidence tracking</li>
            <li>Snapshot capture</li>
            <li>Record & save results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("💡 Tip: Use the Single Image mode for now to classify your photos!")


def render_batch_mode(use_api, api_url):
    """Render batch processing mode"""
    st.markdown("### Batch Processing")
    st.markdown("Upload multiple images for bulk classification")
    
    uploaded_files = st.file_uploader(
        "Choose images...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple images at once"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        if st.button("Process All Images", type="primary", use_container_width=True):
            results = []
            
            progress_text = "Processing images..."
            progress_bar = st.progress(0, text=progress_text)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    if use_api:
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format='JPEG')
                        image_bytes.seek(0)
                        result = predict_with_api(image_bytes, api_url)
                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"]
                    else:
                        if not MODEL_PATH.exists():
                            st.error("❌ Model not found!")
                            break
                        
                        device = torch.device("mps" if torch.backends.mps.is_available() else 
                                            "cuda" if torch.cuda.is_available() else "cpu")
                        
                        if st.session_state.model_loaded is None:
                            st.session_state.model_loaded = load_model_from_checkpoint(str(MODEL_PATH), device)
                        
                        result = predict_local(image, st.session_state.model_loaded, device)
                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"]
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'image': image
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files), 
                                     text=f"Processing {idx + 1}/{len(uploaded_files)}...")
            
            # Display results
            if results:
                st.success(f"✅ Processed {len(results)} images!")
                
                # Summary stats
                df = pd.DataFrame(results)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("🌼 Dandelions", len(df[df['predicted_class'] == 'dandelion']))
                with col3:
                    st.metric("🌿 Grass", len(df[df['predicted_class'] == 'grass']))
                with col4:
                    st.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
                
                st.markdown("---")
                
                # Display results grid
                st.markdown("### 📊 Results")
                
                cols = st.columns(3)
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        st.image(result['image'], use_container_width=True)
                        emoji = CLASS_EMOJIS[result['predicted_class'].capitalize()]
                        st.markdown(f"""
                        <div class='metric-card'>
                            <strong>{emoji} {result['predicted_class'].capitalize()}</strong><br>
                            {result['confidence']*100:.1f}% confidence<br>
                            <small>{result['filename']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Export results
                st.markdown("---")
                st.markdown("### 💾 Export Results")
                
                # Prepare CSV
                export_df = df[['filename', 'predicted_class', 'confidence']].copy()
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("Upload multiple images to start batch processing")


def render_history_mode():
    """Render history and statistics mode"""
    st.markdown("### Prediction History & Statistics")
    
    history = load_history()
    
    if not history:
        st.info("No predictions yet. Start classifying images to see statistics!")
        return
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Statistics overview
    stats = get_history_stats()
    
    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>{stats['total_predictions']}</div>
            <div class='stat-label'>Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>{stats['avg_confidence']*100:.1f}%</div>
            <div class='stat-label'>Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>{stats['dandelion_count']}</div>
            <div class='stat-label'>🌼 Dandelions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>{stats['grass_count']}</div>
            <div class='stat-label'>🌿 Grass</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Timeline chart
    st.markdown("### Prediction Timeline")
    fig = create_history_chart(df, st.session_state.dark_mode)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution pie chart
    st.markdown("### Class Distribution")
    fig = px.pie(
        values=[stats['dandelion_count'], stats['grass_count']],
        names=['Dandelion 🌼', 'Grass 🌿'],
        color_discrete_map={'Dandelion 🌼': CLASS_COLORS['Dandelion'], 'Grass 🌿': CLASS_COLORS['Grass']},
        hole=0.4
    )
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if st.session_state.dark_mode else 'black', size=16)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent predictions table
    st.markdown("### Recent Predictions")
    
    display_df = df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
    display_df['prediction_time'] = display_df['prediction_time'].apply(lambda x: f"{x:.3f}s")
    
    st.dataframe(
        display_df[['timestamp', 'predicted_class', 'confidence', 'prediction_time']].tail(20).iloc[::-1],
        use_container_width=True,
        hide_index=True
    )
    
    # Clear history button
    st.markdown("---")
    if st.button("Clear History", type="secondary", use_container_width=True):
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.rerun()


# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
