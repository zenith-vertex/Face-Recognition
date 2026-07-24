"""OpenCV-based camera adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..core.ports.camera_port import CameraPort

logger = logging.getLogger(__name__)


class OpenCVCameraAdapter(CameraPort):
    """Concrete camera implementation using OpenCV VideoCapture."""

    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        """Initialize the camera adapter.

        Args:
            source: Camera source index (0 for default webcam).
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.source = source
        self.width = width
        self.height = height
        self._capture: np.ndarray | None = None
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Initialize the OpenCV VideoCapture."""
        try:
            import cv2
            self._cv2 = cv2
            self._video_capture = cv2.VideoCapture(self.source)
            self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not self._video_capture.isOpened():
                self._video_capture = None
                logger.warning("Could not open camera")
        except Exception as e:
            self._video_capture = None
            logger.error(f"Failed to initialize camera: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the camera is available."""
        return self._video_capture is not None and self._video_capture.isOpened()

    def capture_frame(self) -> np.ndarray | None:
        """Capture a single frame."""
        if not self.is_available:
            return None
        try:
            ret, frame = self._video_capture.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                return None
            return frame
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None

    def capture_and_save(self, output_path: Path) -> bool:
        """Capture a frame and save it to disk."""
        frame = self.capture_frame()
        if frame is None:
            return False
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._cv2.imwrite(str(output_path), frame)
            return True
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False

    def release(self) -> None:
        """Release camera resources."""
        if self._video_capture and self._video_capture.isOpened():
            self._video_capture.release()
            self._video_capture = None
            logger.debug("Camera released")
