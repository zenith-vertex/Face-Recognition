"""Configuration management for the face recognition system."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self, env_file: Path | None = None) -> None:
        if env_file and env_file.exists():
            load_dotenv(dotenv_path=str(env_file))
        else:
            load_dotenv()

        base_dir = Path(__file__).resolve().parent.parent.parent.parent

        self.app_name: str = os.getenv("APP_NAME", "Face Recognition System")
        self.app_version: str = os.getenv("APP_VERSION", "1.0.0")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        self.database_path: Path = Path(
            os.getenv("DATABASE_PATH", str(base_dir / "data" / "face_recognition.db"))
        )
        self.encodings_dir: Path = Path(
            os.getenv("ENCODINGS_DIR", str(base_dir / "data" / "encodings"))
        )
        self.logs_dir: Path = Path(
            os.getenv("LOGS_DIR", str(base_dir / "data" / "logs"))
        )

        self.camera_source: int = int(
            os.getenv("CAMERA_SOURCE", "0")
        )
        self.camera_width: int = int(
            os.getenv("CAMERA_WIDTH", "640")
        )
        self.camera_height: int = int(
            os.getenv("CAMERA_HEIGHT", "480")
        )

        self.recognition_tolerance: float = float(
            os.getenv("RECOGNITION_TOLERANCE", "0.5")
        )
        self.capture_count: int = int(
            os.getenv("CAPTURE_COUNT", "5")
        )
        self.min_confidence: float = float(
            os.getenv("MIN_CONFIDENCE", "0.5")
        )
        self.export_format: str = os.getenv("EXPORT_FORMAT", "csv")
        self.date_format: str = os.getenv("DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for directory in [
            self.database_path.parent,
            self.encodings_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_database_url(self) -> str:
        """Get the SQLite database URL.

        Returns:
            SQLite connection string.
        """
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.database_path}"
