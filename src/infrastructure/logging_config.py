"""Logging configuration for the face recognition system."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure application logging with Rich and file handlers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file.

    Returns:
        The configured root logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("face_recognition_system")
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(numeric_level)
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    logging.basicConfig(handlers=[stream_handler := logging.StreamHandler(sys.stdout)])
    root = logging.getLogger()
    root.handlers = [stream_handler]
    root.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(numeric_level)

    return logger
