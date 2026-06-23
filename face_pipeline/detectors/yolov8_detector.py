import time
import os
import cv2
import numpy as np
import torch
from .base import BaseDetector


class YOLOv8Detector(BaseDetector):
    def __init__(self, model_path: str = "yolov8n-face.pt"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("Warning: No GPU available, falling back to CPU for YOLOv8")
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            ) from e
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"YOLOv8 model not found at {self.model_path}. "
                f"Download with: powershell -Command \"Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v0.0.0/{self.model_path} -OutFile {self.model_path}\" "
                f"or: curl -L -o {self.model_path} https://github.com/ultralytics/assets/releases/download/v0.0.0/{self.model_path}"
            ) from e
        except Exception as e:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"YOLOv8 model not found at {self.model_path}. "
                    f"Download with: curl -L -o {self.model_path} "
                    f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{self.model_path}"
                ) from e
            raise

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.perf_counter()
        
        results = self.model(frame, verbose=False)
        latency = (time.perf_counter() - start_time) * 1000
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    detections.append({
                        "bbox": (x, y, w, h),
                        "confidence": conf,
                        "latency_ms": latency
                    })
        
        return detections