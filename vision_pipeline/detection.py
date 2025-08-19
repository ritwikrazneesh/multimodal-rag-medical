import torch
from ultralytics import YOLO

class ObjectDetector:
    """YOLO-based object detection for workers and equipment."""
    def __init__(self, model_path='yolov8n.pt'):  # Use pretrained or custom trained model
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Detect objects in a frame."""
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                if box.cls in [0, 2]:  # Class 0: person (worker), class 2: car (equipment)
                    detections.append({
                        'bbox': box.xyxy.tolist(),
                        'conf': box.conf.item(),
                        'class': box.cls.item()
                    })
        return detections