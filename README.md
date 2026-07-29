# Skin Acne Detection System

AI-powered skin acne detection and mapping tool using YOLOv11 object detection model.

## Features

- **Real-time Detection**: Detects acne lesions in skin images using YOLOv11
- **Interactive UI**: Streamlit-based frontend for easy image upload and analysis
- **Visualization**: Displays annotated images with bounding boxes around detected acne
- **Confidence Scoring**: Shows confidence levels for each detection
- **REST API**: FastAPI backend for processing images

## Project Structure

```
skinacne/
├── backend/
│   └── main.py          # FastAPI backend server
├── frontend/
│   └── app.py           # Streamlit frontend application
├── skinacne.ipynb       # Jupyter notebook for model training
├── skinacne.pt          # Trained YOLOv11 model
├── wholemodel.pt        # Alternative trained model file
└── README.md           # This file
```

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. Install required dependencies:

```bash
pip install ultralytics fastapi uvicorn streamlit requests pillow pandas python-multipart
```

2. Ensure the `skinacne.pt` model file is in the root directory.

## Running the Application

### Start the Backend Server

```bash
cd backend
python main.py
```

The backend will start on `http://localhost:8002`

### Start the Frontend Application

```bash
cd frontend
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

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
- **Confidence Threshold**: 0.1 (configurable)

## API Endpoints

### Health Check
```
GET http://localhost:8002/health
```

### Predict
```
POST http://localhost:8002/predict
Content-Type: multipart/form-data
```

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
- Try different confidence thresholds in the backend
- Check if the model is properly loaded

**Connection errors:**
- Verify the backend server is running
- Check the backend URL in the frontend
- Ensure both applications are using the correct ports

## Notes

- The current model is trained for acne detection only (1 class)
- For best results, use high-resolution images with good lighting
- The model may not detect all types of skin conditions - it's specifically trained for acne lesions

## License

This project is for educational and research purposes.
