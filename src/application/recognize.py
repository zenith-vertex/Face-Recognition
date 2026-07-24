"""Use case for recognizing faces from webcam or image."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.domain.face_data import RecognitionEvent
from ..core.domain.user import User
from ..core.ports.camera_port import CameraPort
from ..core.ports.face_recognizer_port import FaceRecognizerPort
from ..core.ports.repositories import (
    FaceEncodingRepository,
    LogRepository,
    UserRepository,
)
from ..core.ports.storage_port import EmbeddingStoragePort

logger = logging.getLogger(__name__)


@dataclass
class RecognizeInput:
    """Input data for face recognition."""

    source: str = "camera"
    image_path: Path | None = None
    camera_source: int = 0
    tolerance: float = 0.5
    save_unknown: bool = True
    output_dir: Path = Path("data/captures")


@dataclass
class RecognizeOutput:
    """Output data from face recognition."""

    recognized_users: list[tuple[User, float]]
    unknown_faces: int
    events_logged: list[RecognitionEvent]
    message: str


class RecognizeFaceUseCase:
    """Use case for recognizing faces from a source."""

    def __init__(
        self,
        user_repo: UserRepository,
        face_encoding_repo: FaceEncodingRepository,
        log_repo: LogRepository,
        face_recognizer: FaceRecognizerPort,
        camera: CameraPort,
        embedding_storage: EmbeddingStoragePort,
    ) -> None:
        self.user_repo = user_repo
        self.face_encoding_repo = face_encoding_repo
        self.log_repo = log_repo
        self.face_recognizer = face_recognizer
        self.camera = camera
        self.embedding_storage = embedding_storage

    def _load_known_encodings(
        self,
    ) -> dict[int, list[np.ndarray]]:
        """Load all known face encodings from storage.

        Returns:
            Dictionary mapping user_id to list of encoding arrays.
        """
        known: dict[int, list[np.ndarray]] = {}
        all_encodings = self.face_encoding_repo.get_all_encodings()

        for face_encoding in all_encodings:
            try:
                encoding = self.embedding_storage.load_encoding(
                    face_encoding.encoding_path
                )
                if face_encoding.user_id not in known:
                    known[face_encoding.user_id] = []
                known[face_encoding.user_id].append(encoding)
            except Exception as e:
                logger.error(
                    f"Failed to load encoding {face_encoding.encoding_path}: {e}"
                )

        return known

    def _process_frame(
        self,
        frame: np.ndarray,
        known_encodings: dict[int, list[np.ndarray]],
        tolerance: float,
        save_unknown: bool,
        output_dir: Path,
    ) -> tuple[list[tuple[User, float]], int, list[RecognitionEvent]]:
        """Process a single frame for face recognition.

        Args:
            frame: The image frame as a numpy array.
            known_encodings: Preloaded known encodings.
            tolerance: Matching tolerance threshold.
            save_unknown: Whether to save unknown face images.
            output_dir: Directory to save unknown faces.

        Returns:
            Tuple of (recognized_users, unknown_count, logged_events).
        """
        temp_path = output_dir / "_temp_frame.jpg"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(temp_path), frame)

        events: list[RecognitionEvent] = []
        recognized: list[tuple[User, float]] = []
        unknown_count = 0

        try:
            face_locations = self.face_recognizer.detect_faces(temp_path)
            if not face_locations:
                return recognized, unknown_count, events

            face_encodings = self.face_recognizer.encode_faces(
                temp_path, face_locations
            )

            for i, (top, right, bottom, left) in enumerate(face_locations):
                if i >= len(face_encodings):
                    continue

                face_encoding = face_encodings[i]
                user_id, confidence = self._find_best_match(
                    known_encodings, face_encoding, tolerance
                )

                if user_id is not None:
                    user = self.user_repo.get_by_id(user_id)
                    if user:
                        recognized.append((user, confidence))
                        event = self.log_repo.log_event(
                            event_type=RecognitionEvent.EVENT_RECOGNIZED,
                            user_id=user.id,
                            confidence=confidence,
                            image_path=temp_path,
                        )
                        events.append(event)
                        logger.info(
                            f"Recognized: {user.name} (confidence: {confidence:.2%})"
                        )
                else:
                    unknown_count += 1
                    if save_unknown:
                        ts = int(time.time() * 1000)
                        unknown_path = (
                            output_dir / f"unknown_{unknown_count}_{ts}.jpg"
                        )
                        face_crop = frame[top:bottom, left:right]
                        cv2.imwrite(str(unknown_path), face_crop)

                    event = self.log_repo.log_event(
                        event_type=RecognitionEvent.EVENT_UNKNOWN,
                        user_id=None,
                        confidence=None,
                        image_path=temp_path,
                    )
                    events.append(event)
                    logger.info("Unknown face detected")

        finally:
            if temp_path.exists():
                temp_path.unlink()

        return recognized, unknown_count, events

    def _find_best_match(
        self,
        known_encodings: dict[int, list[np.ndarray]],
        unknown: np.ndarray,
        tolerance: float,
    ) -> tuple[int | None, float]:
        """Find the best matching user for a face encoding.

        Args:
            known_encodings: Dictionary of user_id to known encodings.
            unknown: The unknown face encoding.
            tolerance: Maximum distance for a match.

        Returns:
            Tuple of (user_id, confidence) or (None, 0.0) if no match.
        """
        best_user_id: int | None = None
        best_distance = tolerance

        for user_id, encodings in known_encodings.items():
            try:
                distances = np.linalg.norm(encodings - unknown, axis=1)
                min_dist = float(np.min(distances))
                if min_dist < best_distance:
                    best_distance = min_dist
                    best_user_id = user_id
            except Exception as e:
                logger.warning(
                    f"Error comparing encodings for user {user_id}: {e}"
                )

        confidence = 1.0 - best_distance if best_user_id is not None else 0.0
        return best_user_id, confidence

    def execute(self, input_data: RecognizeInput) -> RecognizeOutput:
        """Execute face recognition from camera or image.

        Args:
            input_data: Recognition parameters.

        Returns:
            Recognition results.
        """
        known_encodings = self._load_known_encodings()
        if not known_encodings:
            return RecognizeOutput(
                recognized_users=[],
                unknown_faces=0,
                events_logged=[],
                message="No trained encodings found. Run 'train' first.",
            )

        if input_data.source == "camera":
            return self._recognize_from_camera(input_data, known_encodings)
        elif input_data.image_path and input_data.image_path.exists():
            return self._recognize_from_image(input_data, known_encodings)
        else:
            return RecognizeOutput(
                recognized_users=[],
                unknown_faces=0,
                events_logged=[],
                message="No valid source provided (camera or image_path)",
            )

    def _recognize_from_camera(
        self,
        input_data: RecognizeInput,
        known_encodings: dict[int, list[np.ndarray]],
    ) -> RecognizeOutput:
        """Run recognition in a live camera loop."""
        if not self.camera.is_available:
            recognized_users: list[tuple[User, float]] = []
            return RecognizeOutput(
                recognized_users=recognized_users,
                unknown_faces=0,
                events_logged=[],
                message="Camera is not available",
            )

        all_events: list[RecognitionEvent] = []
        total_unknown = 0

        logger.info("Starting live recognition (press 'q' to quit)...")

        window_name = "Face Recognition - Press 'q' to quit"
        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    continue

                recognized, unknown_count, events = self._process_frame(
                    frame, known_encodings, input_data.tolerance,
                    input_data.save_unknown, input_data.output_dir
                )
                all_events.extend(events)
                total_unknown += unknown_count

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cv2.destroyAllWindows()
            self.camera.release()

        message = (
            f"Recognition complete: last frame "
            f"{total_unknown} unknown faces detected"
        )
        return RecognizeOutput(
            recognized_users=[],
            unknown_faces=total_unknown,
            events_logged=all_events,
            message=message,
        )

    def _recognize_from_image(
        self,
        input_data: RecognizeInput,
        known_encodings: dict[int, list[np.ndarray]],
    ) -> RecognizeOutput:
        """Run recognition on a single image file."""
        image_path = input_data.image_path
        frame = cv2.imread(str(image_path))
        if frame is None:
            return RecognizeOutput(
                recognized_users=[],
                unknown_faces=0,
                events_logged=[],
                message=f"Could not load image: {image_path}",
            )

        recognized, unknown_count, events = self._process_frame(
            frame, known_encodings, input_data.tolerance,
            input_data.save_unknown, input_data.output_dir
        )

        message = (
            f"Image recognition complete: {len(recognized)} recognized, "
            f"{unknown_count} unknown faces detected"
        )
        return RecognizeOutput(
            recognized_users=recognized,
            unknown_faces=unknown_count,
            events_logged=events,
            message=message,
        )
