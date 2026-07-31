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
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #9d174d;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #fdf2f8;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ec4899;
        margin-bottom: 1rem;
    }
    .acne-detected {
        color: #be185d;
        font-weight: bold;
    }
    .clear-skin {
        color: #047857;
        font-weight: bold;
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

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2864/2864239.png", width=100)
st.sidebar.title("Configuration")
st.sidebar.markdown("""
### How it works:
1. Upload a high-quality close-up photo of the affected skin area.
2. The image is processed by a YOLOv11 model.
3. Acne lesions are automatically detected and mapped.
""")

# Main page title
st.markdown('<div class="main-title">✨ Skin Acne Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered acne detection and mapping tool.</div>', unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if 'skinacne.pt' exists in the same directory.")
    st.stop()

# Display model information
st.sidebar.markdown(f"""
### Model Information:
- **Model**: YOLOv11
- **Classes**: {model.names}
- **Task**: {model.task}
""")

# ============================================
# FILE UPLOAD AND DISPLAY
# ============================================

# Step 1: Upload image file
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

# Step 2: Check if file is uploaded
if uploaded_file is not None:
    # Step 3: Open and read the image
    image = Image.open(uploaded_file)
    
    # Step 4: Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Step 5: Display original image in left column
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Step 6: Display analysis in right column
    with col2:
        st.subheader("Analysis & Detection")
        
        # Step 7: Run prediction with loading indicator
        with st.spinner("Analyzing skin surface..."):
            try:
                # Step 8: Get predictions from the model
                detections, annotated_image = predict_acne(image, model)
                
                # Step 9: Display the image with detected acne marked
                st.image(annotated_image, use_column_width=True)
                
                # Step 10: Show detection results
                if len(detections) > 0:
                    # Display count of detected acne
                    st.markdown(f'<div class="metric-card">Detections: <span class="acne-detected">Found {len(detections)} acne lesion(s)</span></div>', unsafe_allow_html=True)
                    
                    # Step 11: Create a table with detection details
                    df = pd.DataFrame(detections)
                    
                    # Convert confidence to percentage
                    df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.2f}%")
                    
                    # Round bounding box coordinates
                    df["bbox"] = df["bbox"].apply(lambda x: [round(val, 1) for val in x])
                    
                    # Rename columns for better readability
                    df.rename(columns={
                        "class": "Type",
                        "confidence": "Confidence Score",
                        "bbox": "Bounding Box [xmin, ymin, xmax, ymax]"
                    }, inplace=True)
                    
                    # Display the table
                    st.dataframe(df, use_container_width=True)
                else:
                    # No acne detected
                    st.markdown('<div class="metric-card">Detections: <span class="clear-skin">No active acne lesions detected</span></div>', unsafe_allow_html=True)
                    
            except Exception as e:
                # Handle any errors during analysis
                st.error(f"Error during analysis: {str(e)}")
else:
    # Prompt user to upload an image
    st.info("Please upload a photo of the skin to start the analysis.")
