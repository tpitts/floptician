from __future__ import annotations

import logging

from rich.logging import RichHandler
from rich.text import Text


class BoardLogHandler(RichHandler):
    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        board = getattr(record, "board_text", None)
        if isinstance(board, Text):
            return board.copy()
        return super().render_message(record, message)


def configure_logging(*, debug: bool = False) -> None:
    logging_level = logging.DEBUG if debug else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = BoardLogHandler(
        level=logging_level,
        show_time=True,
        show_path=debug,
        markup=False,
        rich_tracebacks=True,
    )
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ("ultralytics", "obsws_python", "comtypes"):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
