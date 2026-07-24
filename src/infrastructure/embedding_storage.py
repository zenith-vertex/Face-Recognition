"""File system based face embedding storage."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..core.domain.exceptions import StorageError
from ..core.ports.storage_port import EmbeddingStoragePort

logger = logging.getLogger(__name__)


class FileSystemEmbeddingStorage(EmbeddingStoragePort):
    """File system implementation of EmbeddingStoragePort."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize the embedding storage.

        Args:
            base_dir: Base directory for storing encoding files.
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_encoding(self, user_id: int, encoding: np.ndarray) -> Path:
        """Save a face encoding to disk."""
        try:
            encoding_path = self.base_dir / f"user_{user_id}_face.npy"
            np.save(encoding_path, encoding)
            logger.debug(f"Saved encoding to {encoding_path}")
            return encoding_path
        except Exception as e:
            raise StorageError(f"Failed to save encoding: {e}") from e

    def load_encoding(self, path: Path) -> np.ndarray:
        """Load a face encoding from disk."""
        try:
            if not path.exists():
                raise FileNotFoundError(f"Encoding file not found: {path}")
            return np.load(path)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to load encoding {path}: {e}") from e

    def delete_encoding(self, path: Path) -> None:
        """Delete a face encoding from disk."""
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted encoding: {path}")
        except Exception as e:
            raise StorageError(f"Failed to delete encoding {path}: {e}") from e

    def load_all_encodings(self) -> list[tuple[int, Path]]:
        """Load all encoding file paths and their user IDs."""
        results: list[tuple[int, Path]] = []
        try:
            if not self.base_dir.exists():
                return results
            for file_path in self.base_dir.glob("user_*_face*.npy"):
                parts = file_path.stem.split("_")
                if len(parts) >= 2 and parts[0] == "user":
                    try:
                        user_id = int(parts[1])
                        results.append((user_id, file_path))
                    except (ValueError, IndexError):
                        logger.warning(
                            f"Could not parse user ID from filename: {file_path.name}"
                        )
        except Exception as e:
            raise StorageError(f"Failed to scan encodings: {e}") from e
        return results

    def delete_user_encodings(self, user_id: int) -> list[Path]:
        """Delete all encodings for a user."""
        deleted_paths: list[Path] = []
        try:
            for file_path in self.base_dir.glob(f"user_{user_id}_face*.npy"):
                try:
                    file_path.unlink()
                    deleted_paths.append(file_path)
                    logger.debug(f"Deleted encoding: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
        except Exception as e:
            raise StorageError(
                f"Failed to delete encodings for user {user_id}: {e}"
            ) from e
        return deleted_paths
