"""Face encoding and recognition event entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FaceEncoding:
    """Represents a face encoding associated with a user."""

    id: int
    user_id: int
    encoding_path: Path
    image_path: Path
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.user_id <= 0:
            raise ValueError(f"Invalid user_id: {self.user_id}")
        if not self.encoding_path.exists():
            raise FileNotFoundError(f"Encoding file not found: {self.encoding_path}")
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

    def __repr__(self) -> str:
        return (
            f"FaceEncoding(id={self.id}, user_id={self.user_id}, "
            f"image={self.image_path.name})"
        )


@dataclass
class RecognitionEvent:
    """Represents a single recognition event logged by the system."""

    id: int
    event_type: str  # "recognized" or "unknown"
    user_id: int | None
    confidence: float | None
    image_path: Path
    timestamp: datetime = field(default_factory=datetime.now)

    EVENT_RECOGNIZED: str = "recognized"
    EVENT_UNKNOWN: str = "unknown"

    def __post_init__(self) -> None:
        if self.event_type not in (self.EVENT_RECOGNIZED, self.EVENT_UNKNOWN):
            raise ValueError(
                f"Invalid event_type: {self.event_type}. "
                f"Must be '{self.EVENT_RECOGNIZED}' or '{self.EVENT_UNKNOWN}'"
            )
        if self.event_type == self.EVENT_RECOGNIZED and self.user_id is None:
            raise ValueError("user_id is required for recognized events")
        if self.confidence is not None and (self.confidence < 0.0 or self.confidence > 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def __repr__(self) -> str:
        user_info = f"user_id={self.user_id}" if self.user_id else "unknown"
        return (
            f"RecognitionEvent(id={self.id}, type={self.event_type}, "
            f"{user_info}, confidence={self.confidence}, "
            f"timestamp={self.timestamp.isoformat()})"
        )
