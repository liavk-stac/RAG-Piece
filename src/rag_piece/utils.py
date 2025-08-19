"""
Utility functions and logging setup for RAG Piece system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with both console and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name (defaults to timestamp-based name)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"rag_piece_{timestamp}.log"
    
    log_path = logs_dir / log_file
    
    # Create logger
    logger = logging.getLogger("rag_piece")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(levelname)s - %(message)s"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_path}")
    return logger


def validate_input(value: any, expected_type: type, name: str, 
                  min_val: Optional[float] = None, max_val: Optional[float] = None) -> None:
    """
    Validate input parameters with type and range checking.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Parameter name for error messages
        min_val: Minimum allowed value (for numeric types)
        max_val: Maximum allowed value (for numeric types)
    
    Raises:
        ValueError: If validation fails
        TypeError: If type is incorrect
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}")
    
    if isinstance(value, (int, float)) and min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if isinstance(value, (int, float)) and max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")


def safe_file_operation(operation_func, error_msg: str, logger: logging.Logger):
    """
    Safely execute file operations with error handling and logging.
    
    Args:
        operation_func: Function to execute
        error_msg: Error message prefix
        logger: Logger instance
    
    Returns:
        Result of operation_func or None if failed
    """
    try:
        return operation_func()
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}", exc_info=True)
        return None


def count_tokens(text: str) -> int:
    """
    Count tokens in text using simple word splitting.
    
    Args:
        text: Text to count tokens in
    
    Returns:
        Number of tokens
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def slugify(text: str) -> str:
    """
    Convert text to filesystem-safe slug.
    
    Args:
        text: Text to slugify
    
    Returns:
        Slugified text
    """
    if not text:
        return ""
    
    # Replace spaces and special characters
    import re
    slug = re.sub(r'[^\w\s-]', '', text.strip())
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug[:100]  # Limit length
