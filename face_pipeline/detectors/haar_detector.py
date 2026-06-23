import time
import cv2
import numpy as np
from .base import BaseDetector


class HaarDetector(BaseDetector):
    def __init__(self, cascade_path: str = "haarcascade_frontalface_default.xml"):
        self.cascade_path = cascade_path
        self._load_model()

    def _load_model(self):
        try:
            self.classifier = cv2.CascadeClassifier(self.cascade_path)
            if self.classifier.empty():
                raise ValueError(f"Failed to load Haar cascade from {self.cascade_path}")
        except cv2.error as e:
            raise FileNotFoundError(
                f"Haar cascade model not found at {self.cascade_path}. "
                f"Download with: "
                f"curl -o {self.cascade_path} "
                f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{self.cascade_path}"
            ) from e

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        latency = (time.perf_counter() - start_time) * 1000
        for (x, y, w, h) in faces:
            results.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": 1.0,
                "latency_ms": latency
            })
        
        return results