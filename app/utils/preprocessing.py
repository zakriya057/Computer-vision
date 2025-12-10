import os
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime

class ImagePreprocessor:
    def __init__(self, results_folder="results"):
        self.results_folder = results_folder
        # Create results folder if it doesn't exist
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    def preprocess_image(self, file_bytes: bytes) -> np.ndarray:
        """
        Converts raw bytes to an OpenCV-compatible numpy array (BGR format).
        Ready for YOLO or OpenCV processing.
        """
        # Open image with Pillow
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB (ensure consistency)
        image = image.convert("RGB")
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert RGB to BGR (OpenCV standard)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_bgr

    def save_image(self, image: np.ndarray, original_filename: str) -> str:
        """
        Saves the image to the results folder.
        Returns the full path of the saved file.
        """
        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(self.results_folder, filename)
        
        # Save the image using OpenCV
        cv2.imwrite(filepath, image)
        
        return filepath
