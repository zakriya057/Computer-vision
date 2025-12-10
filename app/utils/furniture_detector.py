import numpy as np
import cv2
from ultralytics import YOLO
from typing import List, Dict, Tuple

class FurnitureDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLO model for furniture detection.
        
        Args:
            model_path: Path to YOLO model (downloads automatically if not found)
        """
        # Load the model (downloads automatically if not found)
        self.model = YOLO(model_path)
        
        # COCO Class IDs for furniture items
        # 56: chair, 57: couch, 59: bed, 60: dining table
        self.target_classes = [56, 57, 59, 60]
        
        # COCO class names mapping
        self.class_names = {
            56: "chair",
            57: "couch",
            59: "bed",
            60: "dining table"
        }

    def detect(self, image_bytes: bytes) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect furniture in the image and return bounding boxes with marked image.
        
        Args:
            image_bytes: Raw bytes from the uploaded file
            
        Returns:
            Tuple containing:
                - List of dictionaries with detection info (class_id, class_name, confidence, bbox)
                - Image with bounding boxes drawn
        """
        # 1. Convert bytes to numpy array (image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Run Inference
        results = self.model.predict(
            source=img,
            conf=0.25,  # Minimum confidence to count as a detection
            classes=self.target_classes,  # Filter: ONLY furniture
            verbose=False
        )

        # 3. Extract Boxes and Draw on Image
        detections = []
        marked_img = img.copy()
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            classes = result.boxes.cls.cpu().numpy()  # class IDs
            confidences = result.boxes.conf.cpu().numpy()  # confidence scores
            
            for box, cls_id, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls_id)
                confidence = float(conf)
                
                # Add detection info
                detections.append({
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, "unknown"),
                    "confidence": round(confidence, 2),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })
                
                # Draw bounding box
                cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with class name and confidence
                label = f"{self.class_names.get(class_id, 'unknown')} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(
                    marked_img,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    marked_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

        return detections, marked_img
