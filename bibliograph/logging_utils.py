import logging
import sys

LOGGER_NAME = "bibliograph"


def configure_logging(level: str = "INFO", quiet: bool = False) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.CRITICAL if quiet else getattr(logging, level.upper()))
    logger.propagate = False
    if quiet:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
