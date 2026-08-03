"""Structured logging (NFR-7).

structlog and the stdlib `logging` module are wired into a **single** sink so
that our own events, uvicorn's, and SQLAlchemy's all render identically:

    structlog logger --\
                        >-- ProcessorFormatter --> stderr (console | JSON)
    stdlib logger -----/

Console-rendered and colourised in development, line-delimited JSON in
production. ``request_id`` and other bound context flow into every line via
structlog's contextvars.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import settings

_CONFIGURED = False

# Applied to structlog events *and*, via `foreign_pre_chain`, to records that
# come from stdlib loggers so both paths carry the same keys.
_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.UnicodeDecoder(),
]

_QUIET_LOGGERS = {
    "uvicorn.access": logging.WARNING,  # we emit our own access log
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "sqlalchemy.engine": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
}


def configure_logging(*, level: str | None = None, json_logs: bool | None = None) -> None:
    """Idempotent. Safe to call from app startup, Celery boot, and tests."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or settings.obs.log_level).upper()
    use_json = settings.obs.json_logs if json_logs is None else json_logs

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_SHARED_PROCESSORS,
            # Hands the event dict to the stdlib handler below rather than
            # rendering here — that is what unifies the two pipelines.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=_SHARED_PROCESSORS,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    # uvicorn installs its own handlers with propagate=False before our app is
    # imported. Strip them so its lines render through the sink above too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error", "celery"):
        adopted = logging.getLogger(name)
        adopted.handlers.clear()
        adopted.propagate = True

    for name, noisy_level in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(noisy_level)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)
