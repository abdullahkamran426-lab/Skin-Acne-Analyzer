import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import base64

app = FastAPI(title="Skin Acne Detection API", description="FastAPI backend for detecting skin acne using YOLOv11")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skinacne.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Monkeypatch torch.load to handle PyTorch 2.6+ weights_only=True default unpickling issue
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
    print(f"Model classes: {model.names}")
    print(f"Model task: {model.task}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "YOLOv11", "classes": model.names}

@app.get("/test-model")
def test_model():
    """Test endpoint to verify model is working"""
    try:
        # Create a simple test image (red square on white background)
        import numpy as np
        test_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        test_img[100:200, 100:200] = [255, 0, 0]  # Red square
        
        # Run prediction
        results = model.predict(test_img, conf=0.1, verbose=False)[0]
        
        return {
            "status": "model_tested",
            "detections_count": len(results.boxes),
            "model_classes": model.names,
            "model_task": model.task
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference with very low confidence threshold
        print(f"Image size: {image.size}")
        print(f"Model classes: {model.names}")
        results = model.predict(image, conf=0.1, iou=0.5, imgsz=640, verbose=True)[0]
        print(f"Number of detections: {len(results.boxes)}")
        
        # Parse detections
        detections = []
        boxes = results.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            name = model.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
            detections.append({
                "class": name,
                "confidence": conf,
                "bbox": xyxy
            })
            
        # Get annotated image as base64
        annotated_img_array = results.plot() # returns numpy array in BGR (ultralytics plot uses OpenCV BGR)
        # Convert BGR to RGB
        annotated_image = Image.fromarray(annotated_img_array[..., ::-1])
        
        # Save to memory buffer
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "success": True,
            "detections": detections,
            "image": img_str
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
