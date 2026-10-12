import logging

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAME = "bibliograph"


def configure_logging(level: str = "INFO", quiet: bool = False) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.CRITICAL if quiet else getattr(logging, level.upper()))
    logger.propagate = False
    if quiet:
        return
    handler = RichHandler(
        console=Console(stderr=True),
        show_path=False,
        show_time=False,
        rich_tracebacks=True,
        markup=False,
    )
    logger.addHandler(handler)


def stderr_console() -> Console:
    """Return the configured logging console so live progress can share it."""
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        if isinstance(handler, RichHandler):
            return handler.console
    return Console(stderr=True)


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
