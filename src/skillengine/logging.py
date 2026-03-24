"""
Logging utilities for the skills engine.

Provides a centralized logging configuration for the entire package.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

# Package root logger
_root_logger = logging.getLogger("skillengine")


def setup_logging(
    level: str | int = "INFO",
    format: str | None = None,
    stream: TextIO | None = None,
    file: str | None = None,
) -> None:
    """
    Configure logging for the skills engine.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int
        format: Custom log format string
        stream: Output stream (defaults to stderr)
        file: Optional file path to write logs

    Example:
        from skillengine.logging import setup_logging

        # Basic setup
        setup_logging("DEBUG")

        # With file output
        setup_logging("INFO", file="skills.log")

        # Custom format
        setup_logging("DEBUG", format="%(name)s - %(message)s")
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Set level on root logger
    _root_logger.setLevel(level)

    # Clear existing handlers
    _root_logger.handlers.clear()

    # Default format
    if format is None:
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    formatter = logging.Formatter(format)

    # Stream handler
    stream_handler = logging.StreamHandler(stream or sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    _root_logger.addHandler(stream_handler)

    # File handler (optional)
    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        _root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a submodule.

    Args:
        name: Submodule name (e.g., "engine", "filters")

    Returns:
        Logger instance

    Example:
        from skillengine.logging import get_logger

        logger = get_logger("engine")
        logger.info("Loading skills...")
    """
    if name.startswith("skillengine."):
        return logging.getLogger(name)
    return logging.getLogger(f"skillengine.{name}")


def set_level(level: str | int) -> None:
    """
    Set the log level for the skills engine.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    _root_logger.setLevel(level)


def disable() -> None:
    """Disable all logging for the skills engine."""
    _root_logger.disabled = True


def enable() -> None:
    """Re-enable logging for the skills engine."""
    _root_logger.disabled = False
