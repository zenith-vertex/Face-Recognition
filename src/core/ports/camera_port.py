"""Port interface for camera adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class CameraPort(ABC):
    """Interface for camera capture operations.

    Implementations should wrap OpenCV or similar libraries to provide
    a unified interface for capturing frames from a webcam.
    """

    @abstractmethod
    def capture_frame(self) -> np.ndarray | None:
        """Capture a single frame from the camera.

        Returns:
            The captured frame as a numpy array (BGR format),
            or None if capture failed.
        """
        pass

    @abstractmethod
    def capture_and_save(self, output_path: Path) -> bool:
        """Capture a frame and save it to disk.

        Args:
            output_path: Path where the frame should be saved.

        Returns:
            True if frame was captured and saved successfully.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the camera is available and working.

        Returns:
            True if camera can be accessed, False otherwise.
        """
        pass
