from collections.abc import Generator

import httpx2
import pytest
from conftest import Telemetry
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk.metrics.export import Histogram, Sum
from opentelemetry.trace import SpanKind, StatusCode
from test_watcher import jpeg
from typesafe_sdk import SystemOneResponse, TypeSafeClient, Usage

from frigate_decisions.providers import Jev


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Generator[Jev]:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    jev = Jev("jev-test")
    jev.client.close()

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-resolved",
                "usage": {"input_tokens": 1200, "output_tokens": 8},
                "answers": {"dog": {"type": "noul", "noul": 0.9}},
            },
        )

    jev.client = TypeSafeClient(
        api_key="test-key", model="jev-test", transport=httpx2.MockTransport(respond), compression=False
    )
    yield jev
    jev.client.close()


def test_genai_span_usage_and_metrics(provider: Jev, telemetry: Telemetry) -> None:
    assert provider.ask(jpeg(100), "private context", {"dog": "Private question?"}) == {"dog": 0.9}
    (span,) = telemetry.spans.get_finished_spans()
    assert span.name == "system_one jev-test"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes is not None
    assert span.attributes["gen_ai.provider.name"] == "typesafe"
    assert span.attributes["gen_ai.operation.name"] == "system_one"
    assert span.attributes["gen_ai.request.model"] == "jev-test"
    assert span.attributes["gen_ai.response.model"] == "jev-resolved"
    assert span.attributes["gen_ai.usage.input_tokens"] == 1200
    assert span.attributes["gen_ai.usage.output_tokens"] == 8
    assert span.attributes["server.address"] == "api.typesafe.ai"
    assert span.attributes["server.port"] == 443
    assert "Private" not in str(span.attributes)
    assert not telemetry.logs.get_finished_logs()

    collected = telemetry.collect()
    duration = collected["gen_ai.client.inference.duration"].data
    assert isinstance(duration, Histogram)
    assert duration.data_points[0].count == 1
    assert duration.data_points[0].sum > 0
    tokens = collected["gen_ai.client.inference.usage.input_tokens"].data
    assert isinstance(tokens, Sum)
    assert tokens.data_points[0].value == 1200
    assert tokens.data_points[0].attributes is not None
    assert tokens.data_points[0].attributes["gen_ai.token.modality"] == "unknown"
    assert not any("Private" in str(metric) for metric in collected.values())


@pytest.mark.parametrize(
    ("mode", "on_span", "on_event"),
    [
        ("NO_CONTENT", False, False),
        ("SPAN_ONLY", True, False),
        ("EVENT_ONLY", False, True),
        ("SPAN_AND_EVENT", True, True),
        ("invalid", False, False),
    ],
)
def test_content_capture_modes(
    provider: Jev,
    telemetry: Telemetry,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    on_span: bool,
    on_event: bool,
) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", mode)
    _ = provider.ask(jpeg(100), "a room", {"dog": "Dog on sofa?"})
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes is not None
    assert ("gen_ai.input.messages" in span.attributes) == on_span
    if on_span:
        assert "Dog on sofa?" in str(span.attributes["gen_ai.input.messages"])
        assert "0.9" in str(span.attributes["gen_ai.output.messages"])
        assert isinstance(span.attributes["gen_ai.input.messages"], (list, tuple))
    logs = telemetry.logs.get_finished_logs()
    assert bool(logs) == on_event
    if on_event:
        record = logs[0].log_record
        assert record.event_name == "gen_ai.client.inference.operation.details"
        assert record.severity_number == SeverityNumber.DEBUG
        assert record.trace_id == span.context.trace_id if span.context else False
        assert record.attributes is not None
        assert "Dog on sofa?" in str(record.attributes["gen_ai.input.messages"])
    assert not any("Dog on sofa?" in str(metric) for metric in telemetry.collect().values())


def test_provider_error_marks_span_and_duration_without_error_content(
    provider: Jev,
    telemetry: Telemetry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise TimeoutError("private request content")

    monkeypatch.setattr(provider.client, "system_one", fail)
    with pytest.raises(TimeoutError):
        _ = provider.ask(jpeg(100), None, {"dog": "Dog?"})
    (span,) = telemetry.spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == "TimeoutError"
    assert not span.events
    duration = telemetry.collect()["gen_ai.client.inference.duration"].data
    assert isinstance(duration, Histogram)
    assert duration.data_points[0].attributes is not None
    assert duration.data_points[0].attributes["error.type"] == "TimeoutError"


def test_unknown_token_usage_is_omitted(telemetry: Telemetry) -> None:
    from frigate_decisions.telemetry import inference

    with inference("typesafe", "system_one", "jev-test", "http://localhost:8080") as observed:
        observed.response("jev-resolved", None, None)
        observed.output({"dog": 0.5})
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes is not None
    assert "gen_ai.usage.input_tokens" not in span.attributes
    collected = telemetry.collect()
    assert "gen_ai.client.inference.duration" in collected
    # Delta readers may retain earlier series with zero observations.
    for name, metric in collected.items():
        if name.startswith("gen_ai.client.inference.operation."):
            assert isinstance(metric.data, Histogram)
            assert all(point.count == 0 for point in metric.data.data_points)


def test_failed_call_captures_opted_in_inputs_without_inventing_output(
    provider: Jev,
    telemetry: Telemetry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", "SPAN_AND_EVENT")

    def fail(*args: object, **kwargs: object) -> None:
        raise TimeoutError("request failed")

    monkeypatch.setattr(provider.client, "system_one", fail)
    with pytest.raises(TimeoutError):
        _ = provider.ask(jpeg(100), "a room", {"dog": "Dog on sofa?"})
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes is not None
    assert "Dog on sofa?" in str(span.attributes["gen_ai.input.messages"])
    assert "gen_ai.output.messages" not in span.attributes
    (event,) = telemetry.logs.get_finished_logs()
    assert event.log_record.attributes is not None
    assert event.log_record.attributes["error.type"] == "TimeoutError"
    assert "Dog on sofa?" in str(event.log_record.attributes["gen_ai.input.messages"])
    assert "gen_ai.output.messages" not in event.log_record.attributes


def test_incomplete_answers_preserve_reported_usage(
    provider: Jev,
    telemetry: Telemetry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def incomplete(*args: object, **kwargs: object) -> SystemOneResponse:
        return SystemOneResponse(model="actual", usage=Usage(input_tokens=123, output_tokens=1), answers={})

    monkeypatch.setattr(provider.client, "system_one", incomplete)
    with pytest.raises(KeyError):
        _ = provider.ask(jpeg(100), None, {"dog": "Dog?"})
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["gen_ai.response.model"] == "actual"
    assert span.attributes["gen_ai.usage.input_tokens"] == 123
    assert span.attributes["error.type"] == "KeyError"
    tokens = telemetry.collect()["gen_ai.client.inference.usage.input_tokens"].data
    assert isinstance(tokens, Sum)
    assert sum(point.value for point in tokens.data_points) == 123
