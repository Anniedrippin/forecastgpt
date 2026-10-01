"""
Application-wide logging configuration using loguru.

Every service/tool imports `get_logger(__name__)` instead of calling
`logging.getLogger` directly, so log formatting stays consistent and we have
one place to change verbosity, sinks (file / stdout), or add structured
JSON logs later (e.g. if you plug this into a free observability stack).
"""
import sys

from loguru import logger

from app.core.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    _CONFIGURED = True


def get_logger(name: str):
    configure_logging()
    return logger.bind(module=name)
