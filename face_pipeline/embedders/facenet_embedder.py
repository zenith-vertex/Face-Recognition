import time
import numpy as np
import torch
from .base import BaseEmbedder


class FaceNetEmbedder(BaseEmbedder):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("Warning: No GPU available, falling back to CPU for FaceNet")
        self._load_model()

    def _load_model(self):
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
        except ImportError as e:
            raise ImportError(
                "deepface not installed. Install with: pip install deepface"
            ) from e

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        
        rgb_face = face_crop[:, :, ::-1]  # BGR to RGB
        rgb_face = np.ascontiguousarray(rgb_face)
        
        try:
            result = self.DeepFace.represent(
                rgb_face,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False,
                device=self.device
            )
            
            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)
            else:
                embedding = np.zeros(512, dtype=np.float32)
        except Exception:
            embedding = np.zeros(512, dtype=np.float32)
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding