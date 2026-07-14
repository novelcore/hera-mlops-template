"""Custom colored log formatter for pipeline steps."""

import logging
import sys


class ColorFormatter(logging.Formatter):
    """Log formatter that adds colors and clean alignment."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[34m",  # blue
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }

    LABELS = {
        logging.DEBUG: "DEBUG",
        logging.INFO: " INFO",
        logging.WARNING: " WARN",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "FATAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        label = self.LABELS.get(record.levelno, record.levelname)
        timestamp = self.formatTime(record, "%H:%M:%S")

        name = record.name.split(".")[-1]

        return (
            f"{self.DIM}{timestamp}{self.RESET}  "
            f"{color}{self.BOLD}{label}{self.RESET}  "
            f"{self.DIM}{name:<20}{self.RESET}  "
            f"{record.getMessage()}"
        )


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with the colored formatter."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))
