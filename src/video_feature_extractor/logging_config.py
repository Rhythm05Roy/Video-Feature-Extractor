"""Logging configuration for Video Feature Extractor.

Provides structured logging with configurable levels, file/console handlers,
and optional JSON format for log aggregation systems.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        
        return json.dumps(log_entry)


class ProgressFormatter(logging.Formatter):
    """Formatter that includes progress information when available."""
    
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "progress"):
            progress = record.progress
            return f"[{progress['current']}/{progress['total']}] {record.getMessage()}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    logger_name: str = "video_feature_extractor"
) -> logging.Logger:
    """Configure logging for the video feature extractor.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If None, logs to console only.
        json_format: If True, use JSON formatted output.
        logger_name: Name for the logger.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "video_feature_extractor") -> logging.Logger:
    """Get or create a logger with the specified name.
    
    Args:
        name: Logger name (usually module name).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Context manager for logging progress of long-running operations.
    
    Example:
        with ProgressLogger(logger, "Processing frames", total=1000) as progress:
            for i in range(1000):
                # do work
                progress.update(i + 1)
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        total: int,
        log_interval: int = 100
    ):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.log_interval = log_interval
        self.current = 0
        self.start_time = None
    
    def __enter__(self) -> "ProgressLogger":
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation} ({self.total} items)")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation} - {self.total} items in {elapsed:.2f}s"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation} at item {self.current}/{self.total} "
                f"after {elapsed:.2f}s"
            )
    
    def update(self, current: int, message: str = None) -> None:
        """Update progress and optionally log.
        
        Args:
            current: Current progress count.
            message: Optional message to include in log.
        """
        self.current = current
        
        if current % self.log_interval == 0 or current == self.total:
            pct = (current / self.total) * 100 if self.total > 0 else 0
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = current / elapsed if elapsed > 0 else 0
            
            log_msg = f"{self.operation}: {current}/{self.total} ({pct:.1f}%) - {rate:.1f} items/s"
            if message:
                log_msg += f" - {message}"
            
            self.logger.debug(log_msg)
