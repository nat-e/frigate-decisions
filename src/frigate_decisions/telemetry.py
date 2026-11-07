"""OpenTelemetry instruments. The deployment supplies SDK configuration and exporters."""

import json
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from functools import cache
from urllib.parse import urlsplit

from opentelemetry import metrics, trace
from opentelemetry._logs import SeverityNumber, get_logger
from opentelemetry.util.types import AnyValue

tracer = trace.get_tracer("frigate_decisions")
meter = metrics.get_meter("frigate_decisions")
events = get_logger("frigate_decisions")

frames = meter.create_counter(
    "frigate_decisions.frames", unit="{frame}", description="Frame fetch attempts by outcome."
)
camera_active = meter.create_gauge(
    "frigate_decisions.camera.active", unit="1", description="Whether a camera is asking."
)
last_frame = meter.create_gauge(
    "frigate_decisions.camera.last_frame", unit="s", description="Last fetched frame, Unix time."
)
watcher_errors = meter.create_counter("frigate_decisions.watcher.errors", unit="{error}")
decisions = meter.create_counter(
    "frigate_decisions.decisions", unit="{decision}", description="Answers by threshold outcome."
)
probability = meter.create_histogram(
    "frigate_decisions.decision.probability",
    unit="1",
    description="Probability of yes for each rule.",
    explicit_bucket_boundaries_advisory=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
)
rule_active = meter.create_up_down_counter(
    "frigate_decisions.rule.active", unit="{event}", description="Locally active rule events."
)
rule_events = meter.create_counter(
    "frigate_decisions.rule.events", unit="{event}", description="Rule event starts and end requests."
)
event_duration = meter.create_histogram(
    "frigate_decisions.rule.event.duration",
    unit="s",
    description="First-to-last yes interval for a completed rule event.",
    explicit_bucket_boundaries_advisory=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)
event_operations = meter.create_counter(
    "frigate_decisions.frigate.event.operations",
    unit="{operation}",
    description="Frigate event API attempts by outcome.",
)
pending_ends = meter.create_up_down_counter(
    "frigate_decisions.frigate.event.pending_ends",
    unit="{event}",
    description="Event end requests awaiting a retry.",
)

inference_duration = meter.create_histogram(
    "gen_ai.client.inference.duration",
    unit="s",
    description="GenAI client inference operation duration, including retries.",
    explicit_bucket_boundaries_advisory=(
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ),
)
input_tokens = meter.create_counter("gen_ai.client.inference.usage.input_tokens", unit="{token}")
output_tokens = meter.create_counter("gen_ai.client.inference.usage.output_tokens", unit="{token}")
input_tokens_per_call = meter.create_histogram(
    "gen_ai.client.inference.operation.input_tokens",
    unit="{token}",
    explicit_bucket_boundaries_advisory=(
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
    ),
)
output_tokens_per_call = meter.create_histogram(
    "gen_ai.client.inference.operation.output_tokens",
    unit="{token}",
    explicit_bucket_boundaries_advisory=(
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
    ),
)


@cache
def _capture_content(value: str) -> tuple[bool, bool]:
    modes = {
        "NO_CONTENT": (False, False),
        "SPAN_ONLY": (True, False),
        "EVENT_ONLY": (False, True),
        "SPAN_AND_EVENT": (True, True),
    }
    if value.upper() not in modes:
        logging.getLogger(__name__).warning("Unknown OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT; using NO_CONTENT")
    return modes.get(value.upper(), (False, False))


class Inference:
    """Records response metadata before the caller validates and consumes its answers."""

    def __init__(
        self,
        span: trace.Span,
        attributes: dict[str, str | int],
        details: dict[str, AnyValue],
        capture: tuple[bool, bool],
    ):
        self._span: trace.Span = span
        self._attributes: dict[str, str | int] = attributes
        self._details: dict[str, AnyValue] = details
        self._capture: tuple[bool, bool] = capture

    def response(self, model: str, input_count: int | None, output_count: int | None) -> None:
        self._attributes["gen_ai.response.model"] = model
        self._span.set_attribute("gen_ai.response.model", model)
        for direction, count, counter, histogram in (
            ("input", input_count, input_tokens, input_tokens_per_call),
            ("output", output_count, output_tokens, output_tokens_per_call),
        ):
            if count is not None:
                self._span.set_attribute(f"gen_ai.usage.{direction}_tokens", count)
                self._details[f"gen_ai.usage.{direction}_tokens"] = count
                histogram.record(count, self._attributes)
                if count > 0:
                    counter.add(count, {**self._attributes, "gen_ai.token.modality": "unknown"})

    def output(self, answers: dict[str, float]) -> None:
        if any(self._capture):
            messages: AnyValue = [{"role": "assistant", "parts": [{"type": "text", "content": json.dumps(answers)}]}]
            self._details["gen_ai.output.messages"] = messages
            if self._capture[0]:
                self._span.set_attribute("gen_ai.output.messages", messages)


@contextmanager
def inference(
    provider: str,
    operation: str,
    model: str,
    server_url: str,
    *,
    context: str | None = None,
    questions: dict[str, str] | None = None,
) -> Generator[Inference]:
    """One logical model request, including retries; errors contain only their class name."""
    server = urlsplit(server_url)
    attributes: dict[str, str | int] = {
        "gen_ai.provider.name": provider,
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
        "gen_ai.output.type": "json",
    }
    if server.hostname:
        attributes["server.address"] = server.hostname
        attributes["server.port"] = server.port or (443 if server.scheme == "https" else 80)
    capture = _capture_content(os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", "") or "NO_CONTENT")
    details: dict[str, AnyValue] = {}
    if any(capture):
        # TypeSafe accepts structured state, not chat messages; keep its text in one user message.
        details["gen_ai.input.messages"] = [
            {
                "role": "user",
                "parts": [{"type": "text", "content": json.dumps({"context": context, "questions": questions})}],
            }
        ]
    with tracer.start_as_current_span(
        f"{operation} {model}",
        kind=trace.SpanKind.CLIENT,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        if capture[0]:
            span.set_attributes(details)
        started = time.perf_counter()
        try:
            yield Inference(span, attributes, details, capture)
        except Exception as error:
            attributes["error.type"] = type(error).__qualname__
            span.set_attribute("error.type", attributes["error.type"])
            span.set_status(trace.StatusCode.ERROR)
            raise
        finally:
            inference_duration.record(time.perf_counter() - started, attributes)
            if capture[1]:
                events.emit(
                    event_name="gen_ai.client.inference.operation.details",
                    severity_number=SeverityNumber.DEBUG,
                    attributes={**attributes, **details},
                )
