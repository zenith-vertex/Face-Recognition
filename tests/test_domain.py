"""Tests for domain entities and exceptions."""

from __future__ import annotations

import pytest

from src.face_recognition_system.core.domain.exceptions import (
    CameraUnavailableError,
    DatabaseError,
    UserAlreadyExistsError,
)
from src.face_recognition_system.core.domain.user import User


class TestUser:
    def test_create_user_valid(self) -> None:
        user = User(id=1, name="John Doe")
        assert user.id == 1
        assert user.name == "John Doe"
        assert user.face_count == 0

    def test_user_name_stripped(self) -> None:
        user = User(id=1, name="  John Doe  ")
        assert user.name == "John Doe"

    def test_user_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="User name cannot be empty"):
            User(id=1, name="")

    def test_increment_face_count(self) -> None:
        user = User(id=1, name="John")
        user.increment_face_count()
        assert user.face_count == 1

    def test_decrement_face_count(self) -> None:
        user = User(id=1, name="John", face_count=5)
        user.decrement_face_count()
        assert user.face_count == 4


class TestExceptions:
    def test_user_already_exists(self) -> None:
        exc = UserAlreadyExistsError("User already exists")
        assert exc.message == "User already exists"
        assert isinstance(exc, Exception)

    def test_database_error(self) -> None:
        exc = DatabaseError("DB error")
        assert exc.message == "DB error"

    def test_camera_unavailable(self) -> None:
        exc = CameraUnavailableError("Camera busy")
        assert exc.message == "Camera busy"
