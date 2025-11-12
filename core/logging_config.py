"""
Logging Configuration for ClinOrchestra

Provides structured logging with file rotation, console output, and colored formatting.
Supports multiple log files (main, errors, processing) with automatic rotation.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina
Version: 1.0.0
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# ANSI color codes for console output
class LogColors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED + LogColors.BOLD,
    }

    def format(self, record):
        """Format log record with colors"""
        # Add color to level name
        levelname = record.levelname
        if record.levelno in self.LEVEL_COLORS:
            levelname_colored = (
                f"{self.LEVEL_COLORS[record.levelno]}{levelname}{LogColors.RESET}"
            )
            record.levelname = levelname_colored

        # Format the message
        formatted = super().format(record)

        # Reset levelname for subsequent formatters
        record.levelname = levelname

        return formatted

class StructuredFormatter(logging.Formatter):
    """Formatter that adds structured information to logs"""

    def format(self, record):
        """Add structured fields to log record"""
        # Add timestamp
        record.timestamp = datetime.now().isoformat()

        # Add module hierarchy
        if hasattr(record, 'funcName') and record.funcName:
            record.location = f"{record.module}.{record.funcName}"
        else:
            record.location = record.module

        # Format the message
        return super().format(record)

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_level: str = "INFO",
    file_level: str = "INFO",
    enable_file_logging: bool = True,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Setup logging system for ClinOrchestra

    Args:
        log_dir: Directory for log files
        log_level: Overall logging level
        console_level: Console output level
        file_level: File output level (default INFO)
        enable_file_logging: Whether to log to files
        max_bytes: Maximum log file size before rotation (default 5MB)
        backup_count: Number of backup files to keep (default 3)
        enable_colors: Whether to use colored console output

    Returns:
        Configured root logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console Handler - with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))

    if enable_colors and sys.stdout.isatty():
        console_format = (
            f"{LogColors.DIM}%(asctime)s{LogColors.RESET} | "
            f"%(levelname)s | "
            f"{LogColors.DIM}%(name)s{LogColors.RESET} | "
            f"%(message)s"
        )
        console_formatter = ColoredFormatter(
            console_format,
            datefmt='%H:%M:%S'
        )
    else:
        console_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
        console_formatter = logging.Formatter(console_format, datefmt='%H:%M:%S')

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if enable_file_logging:
        # CRITICAL FIX: Ensure log directory exists before creating handlers
        # This prevents FileNotFoundError during log rotation
        log_path.mkdir(parents=True, exist_ok=True)

        # Main log file - rotating
        main_log_file = log_path / "clinannotate.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        main_handler.setLevel(getattr(logging, file_level.upper()))

        file_format = (
            "%(timestamp)s | %(levelname)-8s | %(location)-40s | "
            "%(message)s [%(pathname)s:%(lineno)d]"
        )
        file_formatter = StructuredFormatter(file_format)
        main_handler.setFormatter(file_formatter)
        root_logger.addHandler(main_handler)

        # Error log file - only errors and above
        error_log_file = log_path / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

        # Processing log file - for extraction pipeline
        processing_log_file = log_path / "processing.log"
        processing_handler = logging.handlers.RotatingFileHandler(
            processing_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        processing_handler.setLevel(logging.INFO)

        # Filter to only include processing-related logs
        class ProcessingFilter(logging.Filter):
            def filter(self, record):
                return 'agent_system' in record.name or 'processing' in record.name or 'extraction' in record.message.lower()

        processing_handler.addFilter(ProcessingFilter())
        processing_handler.setFormatter(file_formatter)
        root_logger.addHandler(processing_handler)

    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    # Log startup message
    root_logger.info("="*80)
    root_logger.info("ClinAnnotate Logging System Initialized")
    root_logger.info(f"Log Directory: {log_path.absolute()}")
    root_logger.info(f"Console Level: {console_level} | File Level: {file_level}")
    root_logger.info("="*80)

    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module

    Ensures the log directory exists before returning the logger.
    This prevents FileNotFoundError if setup_logging() wasn't called.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger instance
    """
    # CRITICAL FIX: Ensure log directory exists
    # This handles cases where get_logger() is called before setup_logging()
    log_dir = Path("logs")
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # If we can't create logs directory, fail gracefully
            # Logger will still work but won't write to files
            pass

    return logging.getLogger(name)

class LogContext:
    """Context manager for temporary log level changes"""

    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize context

        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level

    def __enter__(self):
        """Enter context - change log level"""
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore log level"""
        self.logger.setLevel(self.old_level)
        return False

def log_function_call(func):
    """Decorator to log function entry and exit"""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"→ Entering {func.__name__}()")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"← Exiting {func.__name__}() - Success")
            return result
        except Exception as e:
            logger.error(f"← Exiting {func.__name__}() - Error: {e}", exc_info=True)
            raise

    return wrapper

def log_extraction_stage(stage_name: str, logger: Optional[logging.Logger] = None):
    """Decorator to log extraction pipeline stages"""
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)

            log.info("="*60)
            log.info(f"STAGE: {stage_name}")
            log.info("="*60)

            try:
                result = func(*args, **kwargs)
                log.info(f"✓ {stage_name} completed successfully")
                return result
            except Exception as e:
                log.error(f"✗ {stage_name} failed: {e}", exc_info=True)
                raise

        return wrapper
    return decorator

# Initialize default logging on module import
_default_logger = None

def get_default_logger():
    """Get or create default logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
