"""
VectorLift — Structured JSON Logging
======================================
Provides a ``get_logger`` factory that returns a standard Python Logger whose
handlers emit JSON log records.  Each record includes:

    timestamp   ISO-8601 UTC
    level       DEBUG / INFO / WARNING / ERROR / CRITICAL
    name        Logger name (usually the module)
    message     The log message
    request_id  Current request ID (from context var — empty string if unset)
    + any extra kwargs passed to log calls via the ``extra=`` parameter

Usage
-----
    from core.logging.logger import get_logger, request_id_ctx

    logger = get_logger(__name__)

    # In a FastAPI dependency / middleware:
    request_id_ctx.set("some-uuid")
    logger.info("Processing request", extra={"user_id": "u-123"})

    # Produces:
    # {"timestamp": "2026-04-23T10:00:00.000Z", "level": "INFO",
    #  "name": "my.module", "message": "Processing request",
    #  "request_id": "some-uuid", "user_id": "u-123"}
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, ClassVar, override

# ---------------------------------------------------------------------------
# Context variable — stores the current HTTP request ID
# ---------------------------------------------------------------------------

#: Set this at the start of each request (e.g. in a FastAPI middleware).
#: All log records emitted during that request will include its value.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """
    Formats a LogRecord as a single-line JSON object.

    Standard fields
    ---------------
    * timestamp — ISO-8601 with millisecond precision, always UTC
    * level     — uppercase level name
    * name      — logger name
    * message   — formatted log message
    * request_id — from the ``request_id_ctx`` context variable
    * pid       — process ID (useful in multi-worker deployments)

    Extra fields from ``LogRecord.__dict__`` that are not standard logging
    attributes are included at the top level, making it easy to query by them
    in log-aggregation systems.
    """

    # Standard LogRecord attributes to exclude from the "extra" fields dump
    _STANDARD_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",  # Python 3.12+
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Ensure the message is rendered
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3]
            + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.message,
            "request_id": request_id_ctx.get(""),
            "pid": record.process,
        }

        # Exception info
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exception_text"] = record.exc_text
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Include source location in debug builds
        if record.levelno <= logging.DEBUG:
            payload["location"] = f"{record.pathname}:{record.lineno} ({record.funcName})"

        # Merge extra fields — skip standard attrs to avoid clutter
        for key, val in record.__dict__.items():
            if key not in self._STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = val

        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback: serialize individual fields that fail
            safe_payload = {k: _safe_serialize(v) for k, v in payload.items()}
            return json.dumps(safe_payload, ensure_ascii=False)


def _safe_serialize(value: Any) -> Any:
    """Best-effort JSON serialisation for arbitrary Python objects."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

# Module-level registry — avoids reconfiguring handlers on repeated calls
_configured_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, *, level: int | str | None = None) -> logging.Logger:
    """
    Return a JSON-formatted Logger for the given name.

    Parameters
    ----------
    name:
        Logger name — conventionally ``__name__`` of the calling module.
    level:
        Override the log level for this specific logger.  When ``None``
        (default) the level is determined from ``settings.log_level``.

    Returns
    -------
    logging.Logger
        A fully configured logger that emits JSON to stdout.
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)

    # Determine effective level
    if level is None:
        try:
            from core.config.settings import get_settings

            effective_level = get_settings().log_level.value
        except Exception:
            effective_level = "INFO"
    else:
        effective_level = level if isinstance(level, str) else logging.getLevelName(level)

    logger.setLevel(effective_level)

    # Avoid adding duplicate handlers when get_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        handler.setLevel(effective_level)
        logger.addHandler(handler)

    # Prevent log records from bubbling to the root logger's handlers (which
    # might double-print in some frameworks)
    logger.propagate = False

    _configured_loggers[name] = logger
    return logger


# ---------------------------------------------------------------------------
# Root / application-wide logger
# ---------------------------------------------------------------------------

def configure_root_logging(level: str | None = None) -> None:
    """
    Configure the root logger to emit JSON.

    This is typically called once at application startup (e.g. in
    ``apps/api/main.py`` or a Prefect flow entrypoint).
    """
    try:
        from core.config.settings import get_settings

        effective_level = level or get_settings().log_level.value
    except Exception:
        effective_level = level or "INFO"

    root = logging.getLogger()
    root.setLevel(effective_level)

    # Replace all existing handlers on the root logger
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    handler.setLevel(effective_level)
    root.addHandler(handler)

    # Suppress overly verbose third-party loggers
    for noisy in ("urllib3", "httpx", "elastic_transport", "transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Convenience exception logging helper
# ---------------------------------------------------------------------------

def log_exception(
    logger: logging.Logger,
    msg: str,
    *,
    exc: BaseException | None = None,
    **extra: Any,
) -> None:
    """
    Log an exception with structured extra fields.

    Parameters
    ----------
    logger:  The logger to write to.
    msg:     Human-readable message describing the context.
    exc:     The exception to log.  When ``None`` the current exception
             (if any) from ``sys.exc_info()`` is used.
    **extra: Additional fields to include in the JSON log record.
    """
    exc_info = exc if exc is not None else sys.exc_info()[1]

    tb_str: str | None = None
    if exc_info is not None:
        tb_str = "".join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))

    logger.error(
        msg,
        extra={
            "exception_type": type(exc_info).__name__ if exc_info else None,
            "traceback": tb_str,
            **extra,
        },
    )
