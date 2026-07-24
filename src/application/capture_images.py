"""Use case for capturing face images from a webcam."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.domain.exceptions import CameraUnavailableError, UserNotFoundError
from ..core.domain.user import User
from ..core.ports.camera_port import CameraPort
from ..core.ports.face_recognizer_port import FaceRecognizerPort
from ..core.ports.repositories import FaceEncodingRepository, UserRepository
from ..core.ports.storage_port import EmbeddingStoragePort

logger = logging.getLogger(__name__)


@dataclass
class CaptureImagesInput:
    """Input data for capturing images."""

    user_name: str
    count: int = 5
    display_preview: bool = True
    delay_seconds: float = 0.5
    save_dir: Path = Path("data/captures")


@dataclass
class CaptureImagesOutput:
    """Output data from image capture."""

    user: User
    captured_count: int
    saved_paths: list[Path]
    message: str


class CaptureImagesUseCase:
    """Use case for capturing face images from webcam."""

    def __init__(
        self,
        user_repo: UserRepository,
        face_encoding_repo: FaceEncodingRepository,
        camera: CameraPort,
        face_recognizer: FaceRecognizerPort,
        embedding_storage: EmbeddingStoragePort,
    ) -> None:
        self.user_repo = user_repo
        self.face_encoding_repo = face_encoding_repo
        self.camera = camera
        self.face_recognizer = face_recognizer
        self.embedding_storage = embedding_storage

    def execute(self, input_data: CaptureImagesInput) -> CaptureImagesOutput:
        """Execute the image capture workflow.

        Args:
            input_data: Capture parameters.

        Returns:
            Capture result with user data and saved paths.

        Raises:
            UserNotFoundError: If user does not exist.
            CameraUnavailableError: If camera is not available.
        """
        user = self.user_repo.get_by_name(input_data.user_name)
        if not user:
            raise UserNotFoundError(
                f"User '{input_data.user_name}' not found"
            )

        if not self.camera.is_available:
            raise CameraUnavailableError("Camera is not available or in use")

        input_data.save_dir.mkdir(parents=True, exist_ok=True)

        captured = 0
        saved_paths: list[Path] = []

        logger.info(
            f"Starting capture of {input_data.count} images for user '{user.name}'"
        )

        try:
            while captured < input_data.count:
                frame = self.camera.capture_frame()
                if frame is None:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue

                # Save frame temporarily for face detection
                temp_path = input_data.save_dir / f"temp_{user.id}_{captured}.jpg"
                import cv2
                cv2.imwrite(str(temp_path), frame)

                try:
                    face_locations = self.face_recognizer.detect_faces(temp_path)
                    if face_locations:
                        encodings = self.face_recognizer.encode_faces(
                            temp_path, face_locations
                        )
                        if encodings:
                            encoding = encodings[0]
                            final_name = (
                                input_data.save_dir
                                / f"user_{user.id}_{user.name.replace(' ', '_')}_{captured}_{int(time.time())}.jpg"
                            )
                            import cv2
                            cv2.imwrite(str(final_name), frame)

                            encoding_path = self.embedding_storage.save_encoding(
                                user.id, encoding
                            )

                            self.face_encoding_repo.add_encoding(
                                user_id=user.id,
                                encoding_path=encoding_path,
                                image_path=final_name,
                            )
                            self.user_repo.update_face_count(user.id, 1)

                            captured += 1
                            saved_paths.append(final_name)
                            logger.info(
                                f"Captured face {captured}/{input_data.count} for user '{user.name}'"
                            )
                        else:
                            logger.warning(
                                "Face detected but encoding failed, skipping..."
                            )
                    else:
                        logger.warning(
                            "No face detected in frame, please position face in camera view"
                        )
                finally:
                    if temp_path.exists():
                        temp_path.unlink()

                time.sleep(input_data.delay_seconds)

        finally:
            self.camera.release()

        user.increment_face_count()

        message = (
            f"Successfully captured {captured} face images for user '{user.name}'"
        )
        logger.info(message)
        return CaptureImagesOutput(
            user=user, captured_count=captured, saved_paths=saved_paths, message=message
        )
