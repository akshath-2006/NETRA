"""Logging setup.

Deliberately small. One format, used everywhere, with the module name kept
short so per-camera lines line up::

    14:22:07 | INFO  | video.CAM01   | looping clip (lap 1)
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;203m",
}
_RESET = "\033[0m"


class _Formatter(logging.Formatter):
    def __init__(self, colored: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.colored = colored

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        shown = f"{level:<5}"
        if self.colored:
            shown = f"{_COLORS.get(level, '')}{shown}{_RESET}"
        stamp = self.formatTime(record, self.datefmt)
        return f"{stamp} | {shown} | {record.name:<14} | {record.getMessage()}"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger. Safe to call more than once."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter(colored=sys.stderr.isatty()))
    root.handlers[:] = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
