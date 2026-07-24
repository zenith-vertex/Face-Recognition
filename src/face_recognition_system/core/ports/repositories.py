"""Repository interfaces (ports) for persistence layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from ..domain.face_data import FaceEncoding, RecognitionEvent
from ..domain.user import User


class UserRepository(ABC):
    """Interface for user persistence operations."""

    @abstractmethod
    def create(self, name: str) -> User:
        """Create a new user.

        Args:
            name: The user's full name.

        Returns:
            The created User entity.

        Raises:
            ValueError: If name is empty or invalid.
        """
        pass

    @abstractmethod
    def get_by_id(self, user_id: int) -> User | None:
        """Retrieve a user by their ID.

        Args:
            user_id: The user's unique identifier.

        Returns:
            The User entity if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> User | None:
        """Retrieve a user by their name.

        Args:
            name: The user's name (case-insensitive).

        Returns:
            The User entity if found, None otherwise.
        """
        pass

    @abstractmethod
    def list_all(self) -> list[User]:
        """Retrieve all registered users.

        Returns:
            List of all User entities.
        """
        pass

    @abstractmethod
    def delete(self, user_id: int) -> bool:
        """Delete a user and all associated data.

        Args:
            user_id: The user's unique identifier.

        Returns:
            True if user was deleted, False if not found.
        """
        pass

    @abstractmethod
    def update_face_count(self, user_id: int, delta: int) -> None:
        """Update the face count for a user.

        Args:
            user_id: The user's unique identifier.
            delta: The amount to change the count by (positive or negative).
        """
        pass


class FaceEncodingRepository(ABC):
    """Interface for face encoding persistence operations."""

    @abstractmethod
    def add_encoding(
        self,
        user_id: int,
        encoding_path: Path,
        image_path: Path,
    ) -> FaceEncoding:
        """Add a new face encoding for a user.

        Args:
            user_id: The user's ID.
            encoding_path: Path to the saved .npy encoding file.
            image_path: Path to the original face image.

        Returns:
            The created FaceEncoding entity.
        """
        pass

    @abstractmethod
    def get_by_user_id(self, user_id: int) -> list[FaceEncoding]:
        """Retrieve all face encodings for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            List of FaceEncoding entities.
        """
        pass

    @abstractmethod
    def count_by_user_id(self, user_id: int) -> int:
        """Count the number of face encodings for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            Number of face encodings.
        """
        pass

    @abstractmethod
    def delete_by_user_id(self, user_id: int) -> int:
        """Delete all face encodings for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            Number of encodings deleted.
        """
        pass

    @abstractmethod
    def get_all_encodings(self) -> list[FaceEncoding]:
        """Retrieve all face encodings in the system.

        Returns:
            List of all FaceEncoding entities.
        """
        pass


class LogRepository(ABC):
    """Interface for recognition log persistence operations."""

    @abstractmethod
    def log_event(
        self,
        event_type: str,
        user_id: int | None,
        confidence: float | None,
        image_path: Path,
    ) -> RecognitionEvent:
        """Log a new recognition event.

        Args:
            event_type: "recognized" or "unknown".
            user_id: User ID if recognized, None if unknown.
            confidence: Match confidence (0.0 to 1.0) if known.
            image_path: Path to the captured image.

        Returns:
            The created RecognitionEvent entity.
        """
        pass

    @abstractmethod
    def get_recent(self, limit: int = 50) -> list[RecognitionEvent]:
        """Retrieve recent recognition events.

        Args:
            limit: Maximum number of events to retrieve.

        Returns:
            List of RecognitionEvent entities, most recent first.
        """
        pass

    @abstractmethod
    def get_by_date_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[RecognitionEvent]:
        """Retrieve recognition events within a date range.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            List of RecognitionEvent entities.
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all log entries.

        Returns:
            Number of entries deleted.
        """
        pass
