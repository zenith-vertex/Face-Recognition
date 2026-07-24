"""Face recognition engine using dlib/face_recognition library."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..core.domain.exceptions import (
    ImageProcessingError,
    RecognitionEngineError,
)
from ..core.ports.face_recognizer_port import FaceRecognizerPort

logger = logging.getLogger(__name__)


class FaceRecognitionEngine(FaceRecognizerPort):
    """Concrete implementation of FaceRecognizerPort using face_recognition library."""

    def __init__(self, model: str = "hog") -> None:
        """Initialize the face recognition engine.

        Args:
            model: Detection model to use ('hog' for CPU, 'cnn' for GPU).
        """
        self.model = model
        self._load_engine()

    def _load_engine(self) -> None:
        """Lazy import and validate the face_recognition library."""
        try:
            import face_recognition

            self._face_recognition = face_recognition
        except ImportError as e:
            raise RecognitionEngineError(
                "face_recognition library is not installed. "
                "Install it with: pip install face-recognition"
            ) from e

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image from disk."""
        try:
            image = self._face_recognition.load_image_file(str(image_path))
            return image
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to load image {image_path}: {e}"
            ) from e

    def detect_faces(
        self,
        image_path: Path,
    ) -> list[tuple[int, int, int, int]]:
        """Detect faces in an image."""
        try:
            image = self.load_image(image_path)
            locations = self._face_recognition.face_locations(
                image, model=self.model
            )
            if not locations:
                logger.debug(f"No faces detected in {image_path.name}")
            return locations
        except RecognitionEngineError:
            raise
        except Exception as e:
            raise ImageProcessingError(
                f"Face detection failed for {image_path}: {e}"
            ) from e

    def encode_faces(
        self,
        image_path: Path,
        face_locations: list[tuple[int, int, int, int]] | None = None,
    ) -> list[np.ndarray]:
        """Generate face embeddings."""
        try:
            image = self.load_image(image_path)
            locations = face_locations or self._face_recognition.face_locations(
                image, model=self.model
            )
            encodings = self._face_recognition.face_encodings(image, locations)
            if not encodings:
                logger.debug(f"No face encodings generated for {image_path.name}")
            return encodings
        except RecognitionEngineError:
            raise
        except Exception as e:
            raise ImageProcessingError(
                f"Face encoding failed for {image_path}: {e}"
            ) from e

    def compare_faces(
        self,
        known_encodings: list[np.ndarray],
        unknown_encoding: np.ndarray,
        tolerance: float = 0.6,
    ) -> list[bool]:
        """Compare a face encoding against known encodings."""
        try:
            if not known_encodings:
                return []
            return self._face_recognition.compare_faces(
                known_encodings, unknown_encoding, tolerance=tolerance
            )
        except Exception as e:
            raise RecognitionEngineError(
                f"Face comparison failed: {e}"
            ) from e

    def face_distances(
        self,
        known_encodings: list[np.ndarray],
        unknown_encoding: np.ndarray,
    ) -> list[float]:
        """Compute Euclidean distances between faces."""
        try:
            if not known_encodings:
                return []
            return list(self._face_recognition.face_distance(known_encodings, unknown_encoding))
        except Exception as e:
            raise RecognitionEngineError(
                f"Face distance computation failed: {e}"
            ) from e
