"""User entity representing a registered person in the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class User:
    """Aggregate root representing a registered user with facial data."""

    id: int
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    face_count: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("User name cannot be empty")
        self.name = self.name.strip()

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()

    def increment_face_count(self) -> None:
        """Increment the number of stored face encodings."""
        self.face_count += 1
        self.touch()

    def decrement_face_count(self) -> None:
        """Decrement the number of stored face encodings."""
        if self.face_count > 0:
            self.face_count -= 1
        self.touch()

    def __repr__(self) -> str:
        return (
            f"User(id={self.id}, name='{self.name}', "
            f"faces={self.face_count}, created_at={self.created_at.isoformat()})"
        )
