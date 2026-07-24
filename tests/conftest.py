"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure src package is importable as root package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from infrastructure.config import Config  # noqa: E402


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir: Path) -> Config:
    return Config(
        env_file=None,
    )
