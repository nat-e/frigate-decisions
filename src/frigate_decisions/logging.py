"""One logging pipeline for the app and its dependencies."""

import logging
from datetime import UTC, datetime
from typing import cast

import structlog
from structlog.typing import EventDict


def _record_fields(_logger: object, _method_name: str, event: EventDict) -> EventDict:
    record = cast("logging.LogRecord", event["_record"])
    event["time"] = datetime.fromtimestamp(record.created, UTC).isoformat()
    event["level"] = record.levelname
    event["logger"] = record.name
    for source, target in (("otelTraceID", "trace_id"), ("otelSpanID", "span_id")):
        event.pop(source, None)
        value = record.__dict__.get(source)
        if value and value != "0":
            event[target] = value
    return event


def configure(log_format: str) -> None:
    """Write text or JSON to stderr, including externally injected trace IDs."""
    renderer = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False, event_key="message", timestamp_key="time")
    )
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[structlog.stdlib.ExtraAdder()],
            processors=[
                _record_fields,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.EventRenamer("message"),
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    # Replace basicConfig's console output while preserving external exporters.
    for existing in root.handlers[:]:
        if type(existing) is logging.StreamHandler:
            root.removeHandler(cast("logging.Handler", existing))
            existing.close()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    structlog.configure(
        processors=[structlog.stdlib.filter_by_level, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    for noisy in ("httpx2", "httpcore2", "urllib3", "typesafe_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
