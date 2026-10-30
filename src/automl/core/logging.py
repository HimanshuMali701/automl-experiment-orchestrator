"""Structured logging configuration for AutoML Experiment Orchestrator."""

import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO) -> None:
    """Configure system-wide standard logging format."""
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(level)
        root_logger.addHandler(handler)
    else:
        root_logger.setLevel(level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with the given name."""
    logger_name = f"automl.{name}" if name and not name.startswith("automl") else (name or "automl")
    logger = logging.getLogger(logger_name)
    if not logging.getLogger().handlers:
        configure_logging()
    return logger
