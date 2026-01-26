"""
Logging utilities for cross-platform terminal output.

This module provides utilities for handling Unicode characters in terminal output,
with special consideration for Windows systems that may not support Unicode icons.

Features:
    - Automatic terminal capability detection
    - Fallback to ASCII icons on Windows or unsupported terminals
    - SafeStreamHandler for graceful encoding error handling

Usage:
    from nautical_graph_toolkit.utils.logging_utils import ICONS, SafeStreamHandler

    # Use icons in logging
    logger.info(f"{ICONS['OK']} Operation completed successfully")
    logger.error(f"{ICONS['FAIL']} Operation failed")

    # Use SafeStreamHandler for logging setup
    import logging
    handler = SafeStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
"""

import os
import sys
import logging
from typing import Dict, Optional


def get_status_icons() -> Dict[str, str]:
    """Determine supported status icons based on terminal encoding.

    Prioritizes compatibility over aesthetics:
    1. Windows (os.name == 'nt'): Always use ASCII icons to prevent encoding errors
    2. Test terminal encoding: Use Unicode only if stdout supports it
    3. Fallback: ASCII icons for unsupported terminals

    Returns:
        Dictionary with keys: 'OK', 'FAIL', 'WARN', 'INFO'

    Examples:
        >>> icons = get_status_icons()
        >>> print(f"{icons['OK']} Success")
        [+] Success          # On Windows or unsupported terminals
        ✓ Success           # On supported terminals
    """
    # Force ASCII on Windows to prevent encoding errors with logging handlers
    # Windows console often uses cp1252 or similar encodings that don't support Unicode
    if os.name == 'nt':
        return {
            'OK': '[+]',
            'FAIL': '[x]',
            'WARN': '[!]',
            'INFO': '[i]'
        }

    try:
        # Check if standard output supports unicode characters
        # This catches cases where stdout encoding doesn't support Unicode
        if sys.stdout and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
            '✓'.encode(sys.stdout.encoding)
            return {
                'OK': '✓',
                'FAIL': '✗',
                'WARN': '⚠',
                'INFO': 'ℹ'
            }
    except (UnicodeEncodeError, AttributeError, TypeError):
        pass

    # Fallback for terminals that don't support the characters
    return {
        'OK': '[+]',
        'FAIL': '[x]',
        'WARN': '[!]',
        'INFO': '[i]'
    }


# Module-level constant for easy importing
ICONS = get_status_icons()


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that handles Unicode encoding gracefully.

    This handler wraps the standard StreamHandler with additional error handling
    for Unicode encoding issues, particularly useful on Windows systems.

    The handler attempts to:
    1. Write the message normally
    2. On UnicodeError, encode with 'replace' fallback
    3. On any other error, call handleError() (default behavior)

    Example:
        >>> import logging
        >>> from nautical_graph_toolkit.utils.logging_utils import SafeStreamHandler
        >>>
        >>> logger = logging.getLogger(__name__)
        >>> handler = SafeStreamHandler(sys.stdout)
        >>> handler.setFormatter(logging.Formatter('%(message)s'))
        >>> logger.addHandler(handler)
    """

    def emit(self, record):
        """Emit a log record with graceful Unicode error handling.

        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeError:
            # Fallback: replace problematic characters
            try:
                msg = self.format(record)
                stream = self.stream
                encoding = getattr(stream, 'encoding', 'utf-8') or 'utf-8'
                # Encode with replacement, then decode back to string for write
                msg_safe = msg.encode(encoding, errors='replace').decode(encoding)
                stream.write(msg_safe + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


def get_safe_logger(
    name: str,
    level: int = logging.INFO,
    handler: Optional[logging.Handler] = None
) -> logging.Logger:
    """Create a logger with SafeStreamHandler and icon support.

    Convenience function for creating loggers that handle Unicode gracefully.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: logging.INFO)
        handler: Optional custom handler (default: SafeStreamHandler to stdout)

    Returns:
        Configured logger instance

    Example:
        >>> from nautical_graph_toolkit.utils.logging_utils import get_safe_logger, ICONS
        >>>
        >>> logger = get_safe_logger(__name__)
        >>> logger.info(f"{ICONS['OK']} Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Add handler
    if handler is None:
        handler = SafeStreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# Re-export for convenience
__all__ = [
    'get_status_icons',
    'ICONS',
    'SafeStreamHandler',
    'get_safe_logger'
]