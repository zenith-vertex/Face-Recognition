"""Port interface for face embedding storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingStoragePort(ABC):
    """Interface for face embedding storage operations.

    Implementations handle the serialization and deserialization of
    face embeddings to and from disk.
    """

    @abstractmethod
    def save_encoding(self, user_id: int, encoding: np.ndarray) -> Path:
        """Save a face encoding to disk.

        Args:
            user_id: The ID of the user this encoding belongs to.
            encoding: The 128-dimensional face encoding array.

        Returns:
            Path to the saved encoding file.
        """
        pass

    @abstractmethod
    def load_encoding(self, path: Path) -> np.ndarray:
        """Load a face encoding from disk.

        Args:
            path: Path to the encoding file.

        Returns:
            The 128-dimensional face encoding array.
        """
        pass

    @abstractmethod
    def delete_encoding(self, path: Path) -> None:
        """Delete a face encoding from disk.

        Args:
            path: Path to the encoding file to delete.
        """
        pass

    @abstractmethod
    def load_all_encodings(
        self,
    ) -> list[tuple[int, Path]]:
        """Load all encoding file paths and their associated user IDs.

        Returns:
            List of (user_id, encoding_path) tuples.
        """
        pass

    @abstractmethod
    def delete_user_encodings(self, user_id: int) -> list[Path]:
        """Delete all encodings for a specific user.

        Args:
            user_id: The ID of the user.

        Returns:
            List of deleted file paths.
        """
        pass
