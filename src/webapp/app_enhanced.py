"""
Streamlit Web Application for Plant Classification - Enhanced Version
"""

import io
import time
from pathlib import Path
from PIL import Image
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import plotly.graph_objects as go
import numpy as np


# Configuration
API_URL = "http://localhost:8000"
MODEL_PATH = Path("models/best_model.pth")
CLASS_NAMES = ["Dandelion", "Grass"]
CLASS_EMOJIS = {"Dandelion": "🌼", "Grass": "🌿"}


def create_model(num_classes=2):
    """Create ResNet18 model"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def load_model_from_checkpoint(model_path: str, device: torch.device):
    """Load model from checkpoint"""
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    
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


def create_confidence_chart(probs, class_names):
    """Create confidence bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=probs * 100,
            y=class_names,
            orientation='h',
            marker=dict(
                color=['#FFD700', '#90EE90'],
                line=dict(color='black', width=1.5)
            ),
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Confidence (%)",
        yaxis_title="Class",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    
    return fig


# Page config
st.set_page_config(
    page_title="🌼 Plant Classifier",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 36px !important;
        font-weight: bold;
        text-align: center;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .dandelion-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        border: 3px solid #ffc107;
    }
    .grass-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 3px solid #28a745;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🌼 Dandelion vs Grass Classifier 🌿</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image to identify whether it's a <strong>Dandelion</strong> or <strong>Grass</strong></p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    use_api = st.checkbox("Use API", value=False, help="Use FastAPI backend for predictions")
    
    if use_api:
        api_url = st.text_input("API URL", value=API_URL)
        
        # Test API connection
        if st.button("Test Connection"):
            with st.spinner("Testing..."):
                try:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ API is healthy!")
                    else:
                        st.error(f"❌ API returned status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Cannot connect to API: {str(e)}")
    else:
        st.info("📱 Using local model")
        if MODEL_PATH.exists():
            st.success(f"✅ Model found: {MODEL_PATH}")
        else:
            st.warning(f"⚠️ Model not found at {MODEL_PATH}")
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("""
    - **Architecture**: ResNet18
    - **Classes**: 2 (Dandelion, Grass)
    - **Input Size**: 224x224
    - **Framework**: PyTorch
    """)
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload an image (JPG/PNG)
    2. Wait for classification
    3. View results and confidence
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 12px;'>
        <p>MLOps Project 2025</p>
        <p>Built with Streamlit & PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of dandelion or grass"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Image info
        st.markdown(f"""
        <div class='metric-card'>
            <strong>Image Info</strong><br>
            Size: {image.size[0]}x{image.size[1]}<br>
            Format: {image.format}<br>
            Mode: {image.mode}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### Prediction Result")
    
    if uploaded_file is not None:
        predict_button = st.button("Classify Image", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing image..."):
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
                        
                        with st.spinner("Loading model..."):
                            model = load_model_from_checkpoint(str(MODEL_PATH), device)
                        
                        result = predict_local(image, model, device)
                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"]
                        probs = result["probabilities"]
                    
                    prediction_time = time.time() - start_time
                    
                    # Display prediction
                    pred_class_title = predicted_class.capitalize()
                    emoji = CLASS_EMOJIS.get(pred_class_title, "🌱")
                    
                    st.markdown(f"<p class='big-font'>{emoji} {pred_class_title}</p>", unsafe_allow_html=True)
                    
                    # Result box
                    box_class = "dandelion-box" if predicted_class == "dandelion" else "grass-box"
                    st.markdown(f"""
                    <div class='result-box {box_class}'>
                        <h2 style='text-align: center; margin: 0;'>{emoji} {pred_class_title}</h2>
                        <hr style='margin: 15px 0;'>
                        <p style='text-align: center; font-size: 24px; margin: 10px 0;'>
                            <strong>Confidence: {confidence * 100:.2f}%</strong>
                        </p>
                        <p style='text-align: center; font-size: 16px; color: #666; margin: 5px 0;'>
                            Prediction time: {prediction_time:.3f}s
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                    
                    # Confidence chart
                    st.plotly_chart(
                        create_confidence_chart(probs, CLASS_NAMES),
                        use_container_width=True
                    )
                    
                    # Additional details
                    with st.expander("Detailed Information"):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Dandelion Probability", f"{probs[0]*100:.2f}%")
                        
                        with col_b:
                            st.metric("Grass Probability", f"{probs[1]*100:.2f}%")
                        
                        with col_c:
                            st.metric("Prediction Time", f"{prediction_time:.3f}s")
                        
                        st.markdown("---")
                        st.markdown(f"""
                        **System Information:**
                        - Model: ResNet18
                        - Device: {'GPU (MPS)' if torch.backends.mps.is_available() else 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
                        - Mode: {'API' if use_api else 'Local'}
                        - Image Size: 224x224
                        """)
                    
                    st.success("✅ Classification complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    with st.expander("See error details"):
                        st.exception(e)
    else:
        st.info("Please upload an image to get started")
        
        # Show example images
        st.markdown("### Example Images")
        st.markdown("Try uploading images that look like these:")
        
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.markdown("**🌼 Dandelion**")
            st.markdown("- Yellow flower")
            st.markdown("- Fluffy white seeds")
            st.markdown("- Bright petals")
        
        with ex_col2:
            st.markdown("**🌿 Grass**")
            st.markdown("- Green blades")
            st.markdown("- Lawn or field")
            st.markdown("- Ground cover")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='font-size: 14px; color: #666;'>
        <strong>MLOps Project</strong> - Dandelion vs Grass Classification
    </p>
    <p style='font-size: 12px; color: #999;'>
        Powered by PyTorch, FastAPI, MLflow & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
