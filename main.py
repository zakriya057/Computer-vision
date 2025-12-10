from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from app.utils.preprocessing import ImagePreprocessor
from app.utils.furniture_detector import FurnitureDetector

app = FastAPI(
    title="Computer Vision API",
    description="A FastAPI application for computer vision tasks",
    version="1.0.0"
)

# Initialize Preprocessor and Detector
preprocessor = ImagePreprocessor()
furniture_detector = FurnitureDetector()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Computer Vision API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)) -> Dict:
    """
    Upload an image file, preprocess it for YOLO, and save it.
    
    - **image**: Image file to upload (JPEG, PNG, etc.)
    """
    # 1. Validate File Type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 2. Read the raw bytes
        contents = await image.read()

        # 3. Preprocess the image (Convert to OpenCV format)
        processed_image = preprocessor.preprocess_image(contents)

        # 4. Save the image to results folder
        saved_path = preprocessor.save_image(processed_image, image.filename)

        return {
            "filename": image.filename,
            "content_type": image.content_type,
            "image_shape": processed_image.shape,
            "saved_path": saved_path,
            "message": "Image processed and saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect-furniture")
async def detect_furniture(image: UploadFile = File(...)) -> Dict:
    """
    Detect furniture in an uploaded image using YOLO.
    Returns detected furniture items with bounding boxes and saves the marked image.
    
    - **image**: Image file to upload (JPEG, PNG, etc.)
    """
    # 1. Validate File Type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 2. Read the raw bytes
        contents = await image.read()

        # 3. Detect furniture and get marked image
        detections, marked_image = furniture_detector.detect(contents)

        # 4. Save the marked image to results folder
        marked_filename = f"detected_{image.filename}"
        saved_path = preprocessor.save_image(marked_image, marked_filename)

        return {
            "filename": image.filename,
            "total_detections": len(detections),
            "detections": detections,
            "marked_image_path": saved_path,
            "message": "Furniture detection completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting furniture: {str(e)}")

