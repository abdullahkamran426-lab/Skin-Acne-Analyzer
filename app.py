"""
Skin Acne Detection App - Single Streamlit Application
This app combines UI and backend logic in one file for easy deployment.
"""

import streamlit as st
import os
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import torch

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Skin Acne Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------- Hero ---------- */
    .hero-container {
        background: linear-gradient(135deg, #831843 0%, #db2777 55%, #f472b6 100%);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(157, 23, 77, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #fce7f3;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 1.7rem;
        font-weight: 700;
        color: #9d174d;
        margin: 0.5rem 0 1.2rem 0;
        text-align: center;
    }
    .section-subtext {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: -0.8rem;
        margin-bottom: 1.5rem;
    }

    /* ---------- Workflow cards ---------- */
    .workflow-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #f3d4e4;
        margin-bottom: 0.5rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }
    .workflow-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 28px rgba(157, 23, 77, 0.14);
        border-color: #f9a8d4;
    }
    .step-number {
        background: linear-gradient(135deg, #db2777 0%, #f472b6 100%);
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        box-shadow: 0 6px 14px rgba(236, 72, 153, 0.35);
    }
    .workflow-card h3 {
        font-size: 1.15rem;
        font-weight: 700;
        color: #9d174d;
        margin-bottom: 0.4rem;
    }
    .workflow-card p {
        color: #6b7280;
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* ---------- Upload section ---------- */
    .upload-section {
        background: linear-gradient(180deg, #fdf2f8 0%, #fce7f3 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px dashed #f472b6;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }
    .upload-hint {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* ---------- Results ---------- */
    .results-header {
        font-size: 1.7rem;
        font-weight: 700;
        color: #9d174d;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        text-align: center;
        border-top: 1px solid #f3d4e4;
        padding-top: 1.5rem;
    }
    .image-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #f3d4e4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .image-card-title {
        font-weight: 700;
        color: #9d174d;
        font-size: 1rem;
        margin-bottom: 0.6rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .acne-detected {
        color: #be185d;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .clear-skin {
        color: #047857;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .status-banner {
        padding: 1rem 1.5rem;
        border-radius: 14px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 1.05rem;
    }
    .status-banner.success {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    /* ---------- Model info ---------- */
    .model-info-card {
        background: linear-gradient(180deg, #fdf2f8 0%, #fce7f3 100%);
        padding: 1.8rem;
        border-radius: 20px;
        margin: 2rem 0 0.5rem 0;
        border: 1px solid #f3d4e4;
    }
    .model-info-card h3 {
        font-size: 1.2rem;
        font-weight: 700;
        color: #9d174d;
        margin-bottom: 1rem;
    }
    .model-info-card p {
        color: #4b5563;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD YOLO MODEL
# ============================================
@st.cache_resource
def load_model():
    """
    Load the YOLO model for acne detection.
    This function is cached to avoid reloading the model on every interaction.
    """
    # Fix for PyTorch 2.6+ compatibility
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    
    # Get model path (same directory as this app)
    model_path = os.path.join(os.path.dirname(__file__), "skinacne.pt")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    
    # Load the model
    model = YOLO(model_path)
    return model

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_acne(image, model):
    """
    Run acne detection on the uploaded image.
    
    Args:
        image: PIL Image object
        model: Loaded YOLO model
        
    Returns:
        detections: List of detected acne lesions
        annotated_image: Image with bounding boxes drawn
    """
    # Run prediction with low confidence threshold to catch more detections
    results = model.predict(image, conf=0.1, iou=0.5, imgsz=640, verbose=False)[0]
    
    # Extract detection information
    detections = []
    boxes = results.boxes
    
    for box in boxes:
        # Get class ID and name
        cls_id = int(box.cls[0])
        class_name = model.names.get(cls_id, str(cls_id))
        
        # Get confidence score
        confidence = float(box.conf[0])
        
        # Get bounding box coordinates [xmin, ymin, xmax, ymax]
        bbox = box.xyxy[0].tolist()
        
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })
    
    # Get image with bounding boxes drawn
    annotated_img_array = results.plot()
    annotated_image = Image.fromarray(annotated_img_array[..., ::-1])  # Convert BGR to RGB
    
    return detections, annotated_image

# ============================================
# MAIN APP UI
# ============================================

# ---- Hero Section ----
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ AI-POWERED SKIN ANALYSIS</div>
    <div class="main-title">Skin Acne Analyzer</div>
    <p class="subtitle">AI-Powered Acne Detection and Mapping Tool</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if 'skinacne.pt' exists in the same directory.")
    st.stop()

# ============================================
# FILE UPLOAD AND DISPLAY
# ============================================

st.markdown('<div class="section-header">📤 Analyze Your Skin</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtext">Upload a close-up photo below and let the model map any acne detected</p>', unsafe_allow_html=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)

# Step 1: File uploader widget
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Step 2: Display image if uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Run prediction
    with st.spinner("🔍 Analyzing skin..."):
        detections, annotated_image = predict_acne(image, model)
    
    # Display results
    st.markdown('<div class="results-header">📊 Detection Results</div>', unsafe_allow_html=True)
    
    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-card-title">Original Photo</div>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-card-title">Annotated Photo</div>', unsafe_allow_html=True)
        st.image(annotated_image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if detections:
        pass
    else:
        st.markdown(
            '<div class="status-banner success">✅ <span class="clear-skin">No acne detected</span></div>',
            unsafe_allow_html=True
        )
        st.info("The skin appears clear or no acne was detected above the confidence threshold.")
else:
    st.markdown('<p class="upload-hint">👆 Upload a photo above to begin analysis</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# WORKFLOW SECTION
# ============================================

st.markdown('<div class="section-header">⚙️ How It Works</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">1</div>
        <h3>Upload Photo</h3>
        <p>Upload a close-up photo of the affected skin area</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">2</div>
        <h3>AI Analysis</h3>
        <p>Our YOLOv11 model processes the image to detect acne</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="workflow-card">
        <div class="step-number">3</div>
        <h3>View Results</h3>
        <p>Get instant acne detection with mapping</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MODEL INFORMATION
# ============================================

st.markdown(f"""
<div class="model-info-card">
    <h3>ℹ️ Model Information</h3>
    <p><strong>Model:</strong> YOLOv11</p>
    <p><strong>Classes:</strong> {model.names}</p>
    <p><strong>Task:</strong> {model.task}</p>
</div>
""", unsafe_allow_html=True)