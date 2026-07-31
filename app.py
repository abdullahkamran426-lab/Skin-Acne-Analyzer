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
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #9d174d;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #4b5563;
        margin-bottom: 3rem;
        text-align: center;
    }
    .hero-container {
        background: linear-gradient(135deg, #ec4899 0%, #f472b6 100%);
        padding: 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .workflow-card {
        background-color: #f8fafc;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    .workflow-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .step-number {
        background: linear-gradient(135deg, #ec4899 0%, #f472b6 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .upload-section {
        background-color: #fdf2f8;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #ec4899;
        margin-top: 2rem;
    }
    .acne-detected {
        color: #be185d;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .clear-skin {
        color: #047857;
        font-weight: bold;
        font-size: 1.2rem;
    }
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

# Hero Section
st.markdown('<div class="hero-container">', unsafe_allow_html=True)
st.markdown('<h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem;">✨ Skin Acne Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.3rem; margin-bottom: 0;">AI-Powered Acne Detection and Mapping Tool</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Workflow Section
st.markdown('<h2 style="font-size: 2rem; font-weight: 700; color: #9d174d; margin-bottom: 1.5rem; text-align: center;">How It Works</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="workflow-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-number">1</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size: 1.3rem; font-weight: 700; color: #9d174d; margin-bottom: 0.5rem;">Upload Photo</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #4b5563; margin: 0;">Upload a close-up photo of the affected skin area</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="workflow-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-number">2</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size: 1.3rem; font-weight: 700; color: #9d174d; margin-bottom: 0.5rem;">AI Analysis</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #4b5563; margin: 0;">Our YOLOv11 model processes the image to detect acne</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="workflow-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-number">3</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size: 1.3rem; font-weight: 700; color: #9d174d; margin-bottom: 0.5rem;">View Results</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #4b5563; margin: 0;">Get instant acne detection with mapping</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if 'skinacne.pt' exists in the same directory.")
    st.stop()

# Model Information
st.markdown('<div style="background-color: #fdf2f8; padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">', unsafe_allow_html=True)
st.markdown('<h3 style="font-size: 1.3rem; font-weight: 700; color: #9d174d; margin-bottom: 1rem;">Model Information</h3>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #4b5563; margin: 0.5rem 0;"><strong>Model:</strong> YOLOv11</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #4b5563; margin: 0.5rem 0;"><strong>Classes:</strong> {model.names}</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #4b5563; margin: 0.5rem 0;"><strong>Task:</strong> {model.task}</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FILE UPLOAD AND DISPLAY
# ============================================

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 2rem; font-weight: 700; color: #9d174d; margin-bottom: 1.5rem; text-align: center;">Analyze Your Skin</h2>', unsafe_allow_html=True)

# Step 1: File uploader widget
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Step 2: Display image if uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Photo", use_container_width=True)
    
    # Run prediction
    with st.spinner("Analyzing skin..."):
        detections, annotated_image = predict_acne(image, model)
    
    # Display results
    st.markdown('<h2 style="font-size: 2rem; font-weight: 700; color: #9d174d; margin-top: 2rem; margin-bottom: 1rem;">Detection Results</h2>', unsafe_allow_html=True)
    
    if detections:
        st.markdown(f'<div class="acne-detected">Acne Detected: {len(detections)} lesions</div>', unsafe_allow_html=True)
        
        # Display annotated image
        st.image(annotated_image, caption="Annotated Photo with Detections", use_container_width=True)
        
        # Display detection details
        st.markdown('<h3 style="font-size: 1.5rem; font-weight: 700; color: #9d174d; margin-top: 1.5rem; margin-bottom: 1rem;">Detection Details</h3>', unsafe_allow_html=True)
        df = pd.DataFrame(detections)
        st.dataframe(df, use_container_width=True)
    else:
        st.markdown('<div class="clear-skin">No acne detected</div>', unsafe_allow_html=True)
        st.info("The skin appears clear or no acne was detected above the confidence threshold.")
else:
    st.markdown('<p style="text-align: center; color: #4b5563; font-size: 1.1rem;">👆 Upload a photo above to begin analysis</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
