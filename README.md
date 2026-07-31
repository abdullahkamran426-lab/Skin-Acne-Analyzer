# Skin Acne Detection System

AI-powered skin acne detection and mapping tool using YOLOv11 object detection model.

## Features

- **Real-time Detection**: Detects acne lesions in skin images using YOLOv11
- **Interactive UI**: Streamlit-based interface for easy image upload and analysis
- **Visualization**: Displays annotated images with bounding boxes around detected acne
- **Confidence Scoring**: Shows confidence levels for each detection
- **Simple Deployment**: Single-file application - no separate backend needed

## Project Structure

```
skinacne/
├── app.py               # Main Streamlit application (UI + Backend combined)
├── skinacne.ipynb       # Jupyter notebook for model training
├── skinacne.pt          # Trained YOLOv11 model
└── README.md           # This file
```

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the `skinacne.pt` model file is in the same directory as `app.py`.

## Running the Application

Simply run the Streamlit application:

```bash
streamlit run app.py
```


## Usage

1. Open the Streamlit application in your browser
2. Upload a high-quality close-up photo of the affected skin area
3. The AI will analyze the image and detect acne lesions
4. View the annotated image with detection results
5. Check the confidence scores and bounding box coordinates

## Model Information

- **Model**: YOLOv11 (Nano version)
- **Classes**: 1 class (acne)
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.1 (configurable in code)

## Training the Model

To train your own model, use the provided Jupyter notebook:

```bash
jupyter notebook skinacne.ipynb
```

The notebook includes:
- Dataset preparation
- Model training with optimized hyperparameters
- Evaluation metrics
- Visualization of training results

## Troubleshooting

**No detections found:**
- Ensure the image is clear and well-lit
- Try adjusting the confidence threshold in the `predict_acne()` function
- Check if the model file `skinacne.pt` exists in the same directory

**Model loading errors:**
- Verify all dependencies are installed correctly
- Check that PyTorch is compatible with your system
- Ensure the model file is not corrupted


