import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
from app.core.config import settings


class FaceDetector:
    def __init__(self, model_name: str = None, det_size: Tuple[int, int] = None):
        from insightface.app import FaceAnalysis
        model_name = model_name or settings.INSIGHTFACE_MODEL
        det_size = det_size or (settings.FACE_DETECTION_SIZE, settings.FACE_DETECTION_SIZE)
        self.app = FaceAnalysis(name=model_name)
        ctx_id = 0 if self._has_cuda() else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def _has_cuda(self) -> bool:
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def detect(self, image: np.ndarray) -> List[dict]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "kps": face.kps.astype(float).tolist(),
                "det_score": float(face.det_score),
                "embedding": face.embedding.astype(float).tolist() if face.embedding is not None else None,
            })
        return results

    def detect_and_align(self, image: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            aligned = self._align_face(image, face)
            meta = {
                "bbox": face.bbox.astype(int).tolist(),
                "kps": face.kps.astype(float).tolist(),
                "det_score": float(face.det_score),
                "embedding": face.embedding.astype(float).tolist() if face.embedding is not None else None,
            }
            results.append((aligned, meta))
        return results

    def _align_face(self, image: np.ndarray, face) -> np.ndarray:
        kps = face.kps
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        left_mouth = kps[3]
        right_mouth = kps[4]

        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        mouth_center = ((left_mouth[0] + right_mouth[0]) / 2, (left_mouth[1] + right_mouth[1]) / 2)
        dy = mouth_center[1] - eye_center[1]
        dx = mouth_center[0] - eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center = (int(eye_center[0]), int(eye_center[1]))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        h, w = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = rotated[y1:y2, x1:x2]

        target_size = 112
        try:
            resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized

    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(image)
        if not faces:
            return None
        return faces[0].embedding.astype(np.float32)
