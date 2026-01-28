"""
Logging Configuration for SELTH Experiments

Provides structured logging with:
    - Console output
    - File output with timestamps
    - Experiment tracking
    - Error logging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    experiment_name: str,
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up logging for an experiment.

    Creates:
        - Console handler (INFO level by default)
        - File handler (DEBUG level by default)
        - Formatted output with timestamps
    """
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='[%(levelname)s] %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is None:
        log_dir = Path('./logs')

    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    #Get existing logger by name.
    return logging.getLogger(name)
