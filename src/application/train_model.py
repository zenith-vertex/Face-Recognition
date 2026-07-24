"""Use case for training/updating the face recognition model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..core.ports.repositories import (
    FaceEncodingRepository,
    UserRepository,
)
from ..core.ports.storage_port import EmbeddingStoragePort

logger = logging.getLogger(__name__)


@dataclass
class TrainModelInput:
    """Input data for model training."""

    force: bool = False


@dataclass
class TrainModelOutput:
    """Output data from model training."""

    users_trained: int
    total_encodings: int
    message: str


class TrainModelUseCase:
    """Use case for computing face recognition reference data (centroids)."""

    def __init__(
        self,
        user_repo: UserRepository,
        face_encoding_repo: FaceEncodingRepository,
        embedding_storage: EmbeddingStoragePort,
    ) -> None:
        self.user_repo = user_repo
        self.face_encoding_repo = face_encoding_repo
        self.embedding_storage = embedding_storage

    def execute(self, input_data: TrainModelInput) -> TrainModelOutput:
        """Execute the model training workflow.

        For face_recognition (dlib), training means computing and caching
        reference embeddings/centroids for each user.

        Args:
            input_data: Training parameters.

        Returns:
            Training result summary.
        """
        users = self.user_repo.list_all()
        if not users:
            return TrainModelOutput(
                users_trained=0, total_encodings=0, message="No users registered"
            )

        total_encodings = 0
        users_trained = 0

        for user in users:
            face_encodings = self.face_encoding_repo.get_by_user_id(user.id)
            if not face_encodings:
                logger.warning(
                    f"User '{user.name}' (ID: {user.id}) has no face encodings, skipping"
                )
                continue

            loaded_encodings: list[np.ndarray] = []
            for fe in face_encodings:
                try:
                    encoding = self.embedding_storage.load_encoding(fe.encoding_path)
                    loaded_encodings.append(encoding)
                except Exception as e:
                    logger.error(
                        f"Failed to load encoding {fe.encoding_path}: {e}"
                    )

            if not loaded_encodings:
                logger.warning(
                    f"No valid encodings for user '{user.name}', skipping"
                )
                continue

            stacked = np.stack(loaded_encodings)
            centroid = np.mean(stacked, axis=0)

            centroid_path = (
                self.embedding_storage.base_dir / f"user_{user.id}_centroid.npy"
            )
            centroid_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(centroid_path, centroid)

            total_encodings += len(loaded_encodings)
            users_trained += 1
            logger.info(
                f"Trained model for user '{user.name}' "
                f"with {len(loaded_encodings)} encodings (centroid computed)"
            )

        message = (
            f"Training complete: {users_trained} users processed, "
            f"{total_encodings} total encodings"
        )
        logger.info(message)
        return TrainModelOutput(
            users_trained=users_trained,
            total_encodings=total_encodings,
            message=message,
        )
