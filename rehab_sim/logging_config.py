"""Structured logging setup shared by project entry points."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO", logger_name: str = "rehab") -> logging.Logger:
    """Configure and return a project logger.

    Args:
        level: Standard logging level name, such as ``INFO`` or ``DEBUG``.
        logger_name: Logger namespace to configure.

    Raises:
        ValueError: If *level* is not a standard logging level name.
    """

    normalized_level = level.upper()
    numeric_level = getattr(logging, normalized_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    return logger
