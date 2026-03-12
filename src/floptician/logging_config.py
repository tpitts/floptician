from __future__ import annotations

import logging


def configure_logging(*, debug: bool = False, json_output: bool = False) -> None:
    logging_level = logging.DEBUG if debug else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    if json_output:
        from pythonjsonlogger import json as jsonlogger

        handler = logging.StreamHandler()
        handler.setLevel(logging_level)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(module)s %(lineno)d %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                level=logging_level,
                show_time=True,
                show_path=debug,
                markup=False,
                rich_tracebacks=True,
            )
            root_logger.addHandler(handler)
        except ImportError:
            handler = logging.StreamHandler()
            handler.setLevel(logging_level)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ("ultralytics", "obsws_python", "comtypes"):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
