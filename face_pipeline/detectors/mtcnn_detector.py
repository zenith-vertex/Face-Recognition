import time
import cv2
import numpy as np
import torch
from .base import BaseDetector


class MTCNNDetector(BaseDetector):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("Warning: No GPU available, falling back to CPU for MTCNN")
        self._load_model()

    def _load_model(self):
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                margin=0
            )
        except ImportError as e:
            raise ImportError(
                "facenet-pytorch not installed. Install with: pip install facenet-pytorch"
            ) from e

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.perf_counter()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        
        boxes, probs = self.mtcnn.detect(frame_rgb)
        
        latency = (time.perf_counter() - start_time) * 1000
        results = []
        if boxes is not None:
            for bbox, prob in zip(boxes, probs):
                if prob is not None and prob > 0.5:
                    x, y, x2, y2 = [int(v) for v in bbox]
                    w, h = x2 - x, y2 - y
                    results.append({
                        "bbox": (x, y, w, h),
                        "confidence": float(prob),
                        "latency_ms": latency
                    })
        
        return results