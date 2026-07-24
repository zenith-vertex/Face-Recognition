"""SQLite implementation of repository interfaces."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from ..core.domain.exceptions import DatabaseError
from ..core.domain.face_data import FaceEncoding, RecognitionEvent
from ..core.domain.user import User
from ..core.ports.repositories import (
    FaceEncodingRepository,
    LogRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)


class SQLiteRepository(
    UserRepository, FaceEncodingRepository, LogRepository
):
    """SQLite-based repository implementation for all entities."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection (lazy initialization)."""
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(str(self.database_path))
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
            except sqlite3.Error as e:
                raise DatabaseError(
                    f"Failed to connect to database: {e}"
                )
        return self._conn

    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    face_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS face_encodings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    encoding_path TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL CHECK(event_type IN ('recognized', 'unknown')),
                    user_id INTEGER,
                    confidence REAL,
                    image_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_face_encodings_user_id
                    ON face_encodings(user_id);
                CREATE INDEX IF NOT EXISTS idx_recognition_logs_timestamp
                    ON recognition_logs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_recognition_logs_event_type
                    ON recognition_logs(event_type);
            """)
            conn.commit()
            logger.debug("Database schema initialized")
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to initialize database: {e}"
            )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # UserRepository implementation

    def create(self, name: str) -> User:
        """Create a new user."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO users (name, created_at, updated_at, face_count) VALUES (?, ?, ?, 0)",
                (name, now, now),
            )
            user_id = cursor.lastrowid
            conn.commit()
            return User(id=user_id, name=name, created_at=datetime.fromisoformat(now), updated_at=datetime.fromisoformat(now))
        except sqlite3.IntegrityError:
            raise DatabaseError(f"User '{name}' already exists")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to create user: {e}")

    def get_by_id(self, user_id: int) -> User | None:
        """Retrieve a user by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, created_at, updated_at, face_count FROM users WHERE id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            if row:
                return User(
                    id=row["id"],
                    name=row["name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    face_count=row["face_count"],
                )
            return None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get user: {e}")

    def get_by_name(self, name: str) -> User | None:
        """Retrieve a user by name."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, created_at, updated_at, face_count FROM users WHERE lower(name) = lower(?)",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                return User(
                    id=row["id"],
                    name=row["name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    face_count=row["face_count"],
                )
            return None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get user: {e}")

    def list_all(self) -> list[User]:
        """Retrieve all users."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, created_at, updated_at, face_count FROM users ORDER BY created_at DESC"
            )
            return [
                User(
                    id=row["id"],
                    name=row["name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    face_count=row["face_count"],
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to list users: {e}")

    def delete(self, user_id: int) -> bool:
        """Delete a user by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete user: {e}")

    def update_face_count(self, user_id: int, delta: int) -> None:
        """Update face count for a user."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET face_count = face_count + ?, updated_at = ? WHERE id = ?",
                (delta, datetime.now().isoformat(), user_id),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update face count: {e}")

    # FaceEncodingRepository implementation

    def add_encoding(
        self,
        user_id: int,
        encoding_path: Path,
        image_path: Path,
    ) -> FaceEncoding:
        """Add a face encoding."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO face_encodings (user_id, encoding_path, image_path, created_at) VALUES (?, ?, ?, ?)",
                (user_id, str(encoding_path), str(image_path), now),
            )
            encoding_id = cursor.lastrowid
            conn.commit()
            return FaceEncoding(
                id=encoding_id,
                user_id=user_id,
                encoding_path=encoding_path,
                image_path=image_path,
                created_at=datetime.fromisoformat(now),
            )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to add face encoding: {e}")

    def get_by_user_id(self, user_id: int) -> list[FaceEncoding]:
        """Get all face encodings for a user."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, encoding_path, image_path, created_at FROM face_encodings WHERE user_id = ?",
                (user_id,),
            )
            return [
                FaceEncoding(
                    id=row["id"],
                    user_id=row["user_id"],
                    encoding_path=Path(row["encoding_path"]),
                    image_path=Path(row["image_path"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get face encodings: {e}")

    def count_by_user_id(self, user_id: int) -> int:
        """Count face encodings for a user."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM face_encodings WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            return row["count"] if row else 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to count face encodings: {e}")

    def delete_by_user_id(self, user_id: int) -> int:
        """Delete all face encodings for a user."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM face_encodings WHERE user_id = ?", (user_id,)
            )
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete face encodings: {e}")

    def get_all_encodings(self) -> list[FaceEncoding]:
        """Get all face encodings."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, encoding_path, image_path, created_at FROM face_encodings"
            )
            return [
                FaceEncoding(
                    id=row["id"],
                    user_id=row["user_id"],
                    encoding_path=Path(row["encoding_path"]),
                    image_path=Path(row["image_path"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get all face encodings: {e}")

    # LogRepository implementation

    def log_event(
        self,
        event_type: str,
        user_id: int | None,
        confidence: float | None,
        image_path: Path,
    ) -> RecognitionEvent:
        """Log a recognition event."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO recognition_logs (event_type, user_id, confidence, image_path, timestamp) VALUES (?, ?, ?, ?, ?)",
                (event_type, user_id, confidence, str(image_path), now),
            )
            event_id = cursor.lastrowid
            conn.commit()
            return RecognitionEvent(
                id=event_id,
                event_type=event_type,
                user_id=user_id,
                confidence=confidence,
                image_path=image_path,
                timestamp=datetime.fromisoformat(now),
            )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to log event: {e}")

    def get_recent(self, limit: int = 50) -> list[RecognitionEvent]:
        """Get recent recognition events."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, event_type, user_id, confidence, image_path, timestamp FROM recognition_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [
                RecognitionEvent(
                    id=row["id"],
                    event_type=row["event_type"],
                    user_id=row["user_id"],
                    confidence=row["confidence"],
                    image_path=Path(row["image_path"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get recent logs: {e}")

    def get_by_date_range(
        self, start: datetime, end: datetime
    ) -> list[RecognitionEvent]:
        """Get logs within a date range."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, event_type, user_id, confidence, image_path, timestamp FROM recognition_logs WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC",
                (start.isoformat(), end.isoformat()),
            )
            return [
                RecognitionEvent(
                    id=row["id"],
                    event_type=row["event_type"],
                    user_id=row["user_id"],
                    confidence=row["confidence"],
                    image_path=Path(row["image_path"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get logs by date range: {e}")

    def clear(self) -> int:
        """Clear all logs."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM recognition_logs")
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to clear logs: {e}")
