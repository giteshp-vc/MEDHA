import logging
import re
import sys
from pathlib import Path
from typing import Optional

from config.settings import LOG_LEVEL, LOGS_DIR

# Sensitive pattern regex (e.g. Google API keys starting with AIza, or bearer tokens)
SENSITIVE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z-_]{35}"),
    re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([^'\";\s]+)(['\"]?)", re.IGNORECASE),
]


class CredentialRedactingFormatter(logging.Formatter):
    """Custom formatter to redact API keys and sensitive credentials from log outputs."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        redacted = original
        for pattern in SENSITIVE_PATTERNS:
            if "AIza" in pattern.pattern:
                redacted = pattern.sub("[REDACTED_API_KEY]", redacted)
            else:
                redacted = pattern.sub(r"\1[REDACTED]\3", redacted)
        return redacted


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure structured logging for MEDHA with console and rotating file output."""
    level_name = log_level or LOG_LEVEL
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)

    logger = logging.getLogger("medha")
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Prevent duplicate handlers on re-configuration
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = CredentialRedactingFormatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    # MCP stdio transport reserves stdout for JSON-RPC frames.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    target_file = log_file or (LOGS_DIR / "medha.log")
    try:
        file_handler = logging.FileHandler(target_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Failed to create file handler for {target_file}: {e}\n")

    return logger

