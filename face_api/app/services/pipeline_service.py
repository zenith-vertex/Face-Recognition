import time
import cv2
import numpy as np
from typing import Optional, Tuple

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from face_pipeline.detectors import HaarDetector, MTCNNDetector, YOLOv8Detector
from face_pipeline.embedders import DlibEmbedder, FaceNetEmbedder, ArcFaceEmbedder
from face_pipeline.matcher import match

DETECTORS = {
    "haar": HaarDetector,
    "mtcnn": MTCNNDetector,
    "yolov8": YOLOv8Detector,
}

EMBEDDERS = {
    "dlib": DlibEmbedder,
    "facenet": FaceNetEmbedder,
    "arcface": ArcFaceEmbedder,
}

VECTOR_DIMENSIONS = {
    "dlib": 128,
    "facenet": 512,
    "arcface": 512,
}


class PipelineService:
    def __init__(self, detector_name: str, embedder_name: str, threshold: float = 0.6, margin_ratio: float = 0.15, max_image_dimension: int = 1280):
        self.detector_name = detector_name
        self.embedder_name = embedder_name
        self.threshold = threshold
        self.margin_ratio = margin_ratio
        self.max_image_dimension = max_image_dimension
        self._detector = None
        self._embedder = None

    def initialize(self):
        if self.detector_name not in DETECTORS:
            raise ValueError(f"Unknown detector: {self.detector_name}. Available: {list(DETECTORS.keys())}")
        if self.embedder_name not in EMBEDDERS:
            raise ValueError(f"Unknown embedder: {self.embedder_name}. Available: {list(EMBEDDERS.keys())}")

        self._detector = DETECTORS[self.detector_name]()
        self._embedder = EMBEDDERS[self.embedder_name]()

    def get_detector(self):
        if self._detector is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._detector

    def get_embedder(self):
        if self._embedder is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._embedder

    def validate_embedding_dimension(self, embedding: np.ndarray) -> bool:
        expected_dim = VECTOR_DIMENSIONS.get(self.embedder_name)
        if expected_dim is None:
            raise ValueError(f"Unknown embedder model: {self.embedder_name}")
        return len(embedding) == expected_dim

    def resize_image(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        max_dim = self.max_image_dimension
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)

        if x2 <= x1 or y2 <= y1:
            return frame

        return frame[y1:y2, x1:x2]

    def detect_faces(self, frame: np.ndarray) -> list:
        return self._detector.detect(frame)

    def embed_face(self, face_crop: np.ndarray) -> np.ndarray:
        return self._embedder.embed(face_crop)

    def find_match(self, query_embedding: np.ndarray, known_embeddings: dict) -> dict:
        result = match(query_embedding, known_embeddings, "cosine", self.threshold)
        return result

    def process_frame(self, frame: np.ndarray, known_embeddings: dict, source_camera: str = None) -> list:
        frame = self.resize_image(frame)
        detections = self.detect_faces(frame)
        results = []

        for det in detections:
            bbox = det["bbox"]
            latency_ms = det.get("latency_ms", 0)

            face_crop = self.crop_face(frame, bbox)
            query_emb = self.embed_face(face_crop)

            self.validate_embedding_dimension(query_emb)

            match_result = self.find_match(query_emb, known_embeddings)

            result = {
                "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
                "matched_person_id": None,
                "matched_name": None,
                "confidence": match_result.get("score", 0.0),
                "decision": "match" if match_result["name"] != "Unknown" else "no_match",
                "detect_latency_ms": latency_ms,
                "embed_latency_ms": 0,
                "embedder_used": self.embedder_name,
                "detector_used": self.detector_name,
                "source_camera": source_camera,
            }

            if match_result["name"] != "Unknown":
                result["matched_name"] = match_result["name"]

            results.append(result)

        return results