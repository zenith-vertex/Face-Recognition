"""Port interface for face recognition engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class FaceRecognizerPort(ABC):
    """Interface for face recognition operations.

    Implementations should wrap external libraries like face_recognition (dlib)
    or DeepFace to provide a unified interface for detection, encoding, and comparison.
    """

    @abstractmethod
    def detect_faces(
        self,
        image_path: Path,
    ) -> list[tuple[int, int, int, int]]:
        """Detect faces in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of face locations as (top, right, bottom, left) tuples.
            Returns empty list if no faces detected.
        """
        pass

    @abstractmethod
    def encode_faces(
        self,
        image_path: Path,
        face_locations: list[tuple[int, int, int, int]] | None = None,
    ) -> list[np.ndarray]:
        """Generate face embeddings (encodings) for detected faces.

        Args:
            image_path: Path to the image file.
            face_locations: Optional pre-detected face locations.

        Returns:
            List of 128-dimensional numpy arrays, one per face.
        """
        pass

    @abstractmethod
    def compare_faces(
        self,
        known_encodings: list[np.ndarray],
        unknown_encoding: np.ndarray,
        tolerance: float = 0.6,
    ) -> list[bool]:
        """Compare a face encoding against a list of known encodings.

        Args:
            known_encodings: List of known face encoding arrays.
            unknown_encoding: The face encoding to compare.
            tolerance: Maximum distance for a match (lower = stricter).

        Returns:
            List of booleans indicating matches for each known encoding.
        """
        pass

    @abstractmethod
    def face_distances(
        self,
        known_encodings: list[np.ndarray],
        unknown_encoding: np.ndarray,
    ) -> list[float]:
        """Compute Euclidean distances between a face and known encodings.

        Args:
            known_encodings: List of known face encoding arrays.
            unknown_encoding: The face encoding to compare.

        Returns:
            List of float distances (lower = more similar).
        """
        pass

    @abstractmethod
    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            Image as a numpy array (RGB format).
        """
        pass
