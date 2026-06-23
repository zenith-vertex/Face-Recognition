import time
import numpy as np
import face_recognition
from .base import BaseEmbedder


class DlibEmbedder(BaseEmbedder):
    def __init__(self):
        try:
            import dlib
            self.dlib = dlib
        except ImportError as e:
            raise ImportError(
                "dlib not installed. Install with: pip install dlib face_recognition. "
                "Note: dlib requires cmake and a C++ compiler. On Windows: "
                "pip install cmake; On Mac: brew install cmake"
            ) from e

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(128, dtype=np.float32)
        
        rgb_face = face_crop[:, :, ::-1]  # BGR to RGB
        
        encodings = face_recognition.face_encodings(rgb_face)
        
        if len(encodings) == 0:
            return np.zeros(128, dtype=np.float32)
        
        embedding = encodings[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)