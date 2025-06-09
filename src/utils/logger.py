"""
Logging configuration for the NBA prediction model.
"""
import logging
import sys
from pathlib import Path

from .config import LOG_LEVEL, LOG_FORMAT

def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Set up a logger with the specified name and optional log file.
    
    Args:
        name: Name of the logger
        log_file: Optional path to log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 