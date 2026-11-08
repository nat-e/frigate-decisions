import json
import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast, override

import pytest
import structlog

from frigate_decisions import __main__
from frigate_decisions.config import Config
from frigate_decisions.logging import configure


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    settings = structlog.get_config()
    factory = logging.getLogRecordFactory()
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers = handlers
    root.setLevel(level)
    structlog.configure(**settings)  # pyright: ignore[reportAny] structlog's config snapshot is untyped
    logging.setLogRecordFactory(factory)


def test_json_fields_and_dependency_logs(capsys: pytest.CaptureFixture[str]) -> None:
    configure("json")
    structlog.stdlib.get_logger("app").bind(camera="litter").info(
        "rule started", rule="gaia_visit", event_id="evt1", probability=0.93
    )
    logging.getLogger("dependency").warning("request %s failed", "123", extra={"camera": "door"})
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    event = cast("dict[str, object]", json.loads(lines[0]))
    timestamp = datetime.fromisoformat(cast("str", event.pop("time")))
    assert timestamp.tzinfo == UTC
    assert event == {
        "level": "INFO",
        "logger": "app",
        "message": "rule started",
        "camera": "litter",
        "rule": "gaia_visit",
        "event_id": "evt1",
        "probability": 0.93,
    }
    dependency = cast("dict[str, object]", json.loads(lines[1]))
    assert dependency["message"] == "request 123 failed"
    assert dependency["level"] == "WARNING"
    assert dependency["logger"] == "dependency"
    assert dependency["camera"] == "door"
    assert "rule" not in dependency


def test_json_exception_stays_on_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure("json")
    try:
        raise ValueError("bad\nvalue")
    except ValueError:
        structlog.stdlib.get_logger("app").exception("unexpected\nerror", camera="litter")
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    event = cast("dict[str, object]", json.loads(lines[0]))
    assert event["message"] == "unexpected\nerror"
    assert "ValueError: bad\nvalue" in cast("str", event["exception"])
    assert "trace_id" not in event
    assert "span_id" not in event


def test_injected_trace_ids_with_preconfigured_logging(capsys: pytest.CaptureFixture[str]) -> None:
    original = logging.getLogRecordFactory()

    def correlated_record(*args: object, **kwargs: object) -> logging.LogRecord:
        record = original(*args, **kwargs)
        record.__dict__.update(otelTraceID="a" * 32, otelSpanID="b" * 16)
        return record

    logging.setLogRecordFactory(correlated_record)
    logging.basicConfig(format="preconfigured: %(message)s", force=True)
    configure("json")
    structlog.stdlib.get_logger("app").info("decision", probability=0.7)
    logging.getLogger("dependency").warning("request failed")
    for line in capsys.readouterr().err.splitlines():
        event = cast("dict[str, object]", json.loads(line))
        assert event["trace_id"] == "a" * 32
        assert event["span_id"] == "b" * 16
        assert "otelTraceID" not in event
        assert "otelSpanID" not in event


def test_external_exporter_is_preserved(capsys: pytest.CaptureFixture[str]) -> None:
    class Exporter(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        @override
        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    exporter = Exporter()
    logging.basicConfig(handlers=[logging.StreamHandler(), exporter], force=True)
    configure("json")
    structlog.stdlib.get_logger("app").info("decision", camera="litter", rule="gaia_visit", probability=0.7)
    logging.getLogger("dependency").warning("request failed")
    assert len(exporter.records) == 2
    body = cast("dict[str, object]", exporter.records[0].msg)
    assert body["event"] == "decision"
    assert body["camera"] == "litter"
    assert body["rule"] == "gaia_visit"
    assert body["probability"] == 0.7
    assert len(capsys.readouterr().err.splitlines()) == 2


@pytest.mark.parametrize("mode", ["text", "json"])
def test_level_filtering(mode: str, capsys: pytest.CaptureFixture[str]) -> None:
    configure(mode)
    log = structlog.stdlib.get_logger("app")
    log.debug("hidden debug")
    logging.getLogger("httpx2").info("hidden request")
    log.info("visible", camera="litter")
    output = capsys.readouterr().err
    assert "hidden" not in output
    assert "visible" in output
    assert "litter" in output
    assert "\x1b" not in output


@pytest.mark.parametrize(
    "env, args, expected",
    [
        (None, [], "text"),
        ("json", [], "json"),
        ("json", ["--log-format", "text"], "text"),
        ("invalid", ["--log-format", "json"], "json"),
    ],
)
def test_cli_format_selection(env: str | None, args: list[str], expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if env is None:
        monkeypatch.delenv("FRIGATE_DECISIONS_LOG_FORMAT", raising=False)
    else:
        monkeypatch.setenv("FRIGATE_DECISIONS_LOG_FORMAT", env)
    monkeypatch.setattr(sys, "argv", ["frigate-decisions", *args])
    selected: list[str] = []
    monkeypatch.setattr(__main__, "configure", selected.append)

    def load_config(path: Path) -> Config:
        return Config("http://frigate:5000", "jev", None, ())

    def make_provider(name: str, model: str | None) -> object:
        return object()

    monkeypatch.setattr(__main__, "load", load_config)
    monkeypatch.setattr(__main__, "make_provider", make_provider)
    __main__.main()
    assert selected == [expected]


@pytest.mark.parametrize("args", [[], ["--log-format", "invalid"]])
def test_invalid_format_fails_before_loading_config(
    args: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FRIGATE_DECISIONS_LOG_FORMAT", "invalid")
    monkeypatch.setattr(sys, "argv", ["frigate-decisions", *args])
    with pytest.raises(SystemExit) as exit_info:
        __main__.main()
    assert exit_info.value.code == 2
    assert "invalid" in capsys.readouterr().err
