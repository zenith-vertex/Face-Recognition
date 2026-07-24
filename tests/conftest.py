"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.face_recognition_system.infrastructure.config import Config


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir: Path) -> Config:
    return Config(
        env_file=None,
    )
