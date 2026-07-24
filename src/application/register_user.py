"""Use case for registering a new user in the system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..core.domain.exceptions import (
    InsufficientFaceDataError,
    UserAlreadyExistsError,
)
from ..core.domain.user import User
from ..core.ports.face_recognizer_port import FaceRecognizerPort
from ..core.ports.repositories import FaceEncodingRepository, UserRepository
from ..core.ports.storage_port import EmbeddingStoragePort

logger = logging.getLogger(__name__)


@dataclass
class RegisterUserInput:
    """Input data for registering a user."""

    name: str
    image_paths: list[Path]
    min_faces_required: int = 1


@dataclass
class RegisterUserOutput:
    """Output data from user registration."""

    user: User
    faces_registered: int
    message: str


class RegisterUserUseCase:
    """Use case for registering a new user with face data."""

    def __init__(
        self,
        user_repo: UserRepository,
        face_encoding_repo: FaceEncodingRepository,
        face_recognizer: FaceRecognizerPort,
        embedding_storage: EmbeddingStoragePort,
    ) -> None:
        self.user_repo = user_repo
        self.face_encoding_repo = face_encoding_repo
        self.face_recognizer = face_recognizer
        self.embedding_storage = embedding_storage

    def execute(self, input_data: RegisterUserInput) -> RegisterUserOutput:
        """Execute the user registration workflow.

        Args:
            input_data: Registration parameters.

        Returns:
            Registration result with user data.

        Raises:
            UserAlreadyExistsError: If user with same name already exists.
            InsufficientFaceDataError: If too few valid face embeddings.
        """
        existing = self.user_repo.get_by_name(input_data.name)
        if existing:
            raise UserAlreadyExistsError(
                f"User '{input_data.name}' already exists (ID: {existing.id})"
            )

        user = self.user_repo.create(input_data.name)
        logger.info(f"Created new user: {user}")

        total_faces = 0
        for image_path in input_data.image_paths:
            if not image_path.exists():
                logger.warning(f"Image not found, skipping: {image_path}")
                continue

            try:
                face_locations = self.face_recognizer.detect_faces(image_path)
                if not face_locations:
                    logger.warning(f"No faces detected in {image_path.name}")
                    continue

                encodings = self.face_recognizer.encode_faces(
                    image_path, face_locations
                )
                if not encodings:
                    logger.warning(
                        f"Could not generate encodings for {image_path.name}"
                    )
                    continue

                encoding = encodings[0]
                encoding_path = self.embedding_storage.save_encoding(
                    user_id=user.id,
                    encoding=encoding,
                    suffix=f"_{total_faces}",
                )

                self.face_encoding_repo.add_encoding(
                    user_id=user.id,
                    encoding_path=encoding_path,
                    image_path=image_path,
                )
                self.user_repo.update_face_count(user.id, 1)

                total_faces += 1
                logger.debug(
                    f"Registered face {total_faces} for user {user.id} from {image_path.name}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing image {image_path}: {e}", exc_info=True
                )
                continue

        if total_faces < input_data.min_faces_required:
            self.user_repo.delete(user.id)
            raise InsufficientFaceDataError(
                f"Only {total_faces} valid faces found, "
                f"minimum required: {input_data.min_faces_required}"
            )

        message = (
            f"Successfully registered user '{user.name}' (ID: {user.id}) "
            f"with {total_faces} face encoding(s)"
        )
        logger.info(message)
        return RegisterUserOutput(
            user=user, faces_registered=total_faces, message=message
        )
