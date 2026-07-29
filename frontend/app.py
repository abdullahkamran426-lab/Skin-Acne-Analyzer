import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd

# Set page config for theme and title
st.set_page_config(
    page_title="Skin Acne Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
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

# Sidebar UI
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2864/2864239.png", width=100)
st.sidebar.title("Configuration")
backend_url = "http://localhost:8002"
st.sidebar.markdown("""
### How it works:
1. Upload a high-quality close-up photo of the affected skin area.
2. The image is processed by a YOLOv11 model.
3. Acne lesions are automatically detected and mapped.
""")

# Main page content
st.markdown('<div class="main-title">✨ Skin Acne Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered acne detection and mapping tool.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Analysis & Detection")
        
        # Call backend api
        with st.spinner("Analyzing skin surface..."):
            try:
                # Prepare payload
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or 'JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                response = requests.post(f"{backend_url}/predict", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("success"):
                        detections = result.get("detections", [])
                        
                        # Display annotated image
                        annotated_image_base64 = result.get("image")
                        annotated_image_bytes = base64.b64decode(annotated_image_base64)
                        annotated_image = Image.open(io.BytesIO(annotated_image_bytes))
                        
                        st.image(annotated_image, use_column_width=True)
                        
                        # Display metrics
                        if len(detections) > 0:
                            st.markdown(f'<div class="metric-card">Detections: <span class="acne-detected">Found {len(detections)} acne lesion(s)</span></div>', unsafe_allow_html=True)
                            
                            # Create DataFrame for detections
                            df = pd.DataFrame(detections)
                            df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.2f}%")
                            df["bbox"] = df["bbox"].apply(lambda x: [round(val, 1) for val in x])
                            df.rename(columns={"class": "Type", "confidence": "Confidence Score", "bbox": "Bounding Box [xmin, ymin, xmax, ymax]"}, inplace=True)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.markdown('<div class="metric-card">Detections: <span class="clear-skin">No active acne lesions detected</span></div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Error during backend analysis: {result.get('error')}")
                else:
                    st.error(f"Could not connect to backend server. Status Code: {response.status_code}. Make sure backend is running at {backend_url}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}. Please check backend API URL in the sidebar.")
else:
    st.info("Please upload a photo of the skin to start the analysis.")
