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
    explicit_bucket_boundaries_advisory=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536),
)
output_tokens_per_call = meter.create_histogram(
    "gen_ai.client.inference.operation.output_tokens",
    unit="{token}",
    explicit_bucket_boundaries_advisory=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536),
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
    """Records a provider response without changing the provider's return type."""

    def __init__(self, span: trace.Span, attributes: dict[str, str | int]):
        self._span: trace.Span = span
        self._attributes: dict[str, str | int] = attributes
        self._capture_span: bool
        self._capture_event: bool
        self._capture_span, self._capture_event = _capture_content(
            os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", "NO_CONTENT") or "NO_CONTENT"
        )

    def response(
        self,
        model: str,
        input_count: int | None,
        output_count: int | None,
        context: str | None,
        questions: dict[str, str],
        answers: dict[str, float],
    ) -> None:
        self._attributes["gen_ai.response.model"] = model
        self._span.set_attribute("gen_ai.response.model", model)
        usage: dict[str, AnyValue] = {}
        for direction, count, counter, histogram in (
            ("input", input_count, input_tokens, input_tokens_per_call),
            ("output", output_count, output_tokens, output_tokens_per_call),
        ):
            if count is not None:
                self._span.set_attribute(f"gen_ai.usage.{direction}_tokens", count)
                usage[f"gen_ai.usage.{direction}_tokens"] = count
                histogram.record(count, self._attributes)
                if count > 0:
                    counter.add(count, {**self._attributes, "gen_ai.token.modality": "unknown"})

        if self._capture_span or self._capture_event:
            # TypeSafe accepts structured state, not chat messages; keep its text in one user message.
            content: dict[str, AnyValue] = {
                "gen_ai.input.messages": [
                    {
                        "role": "user",
                        "parts": [
                            {"type": "text", "content": json.dumps({"context": context, "questions": questions})}
                        ],
                    }
                ],
                "gen_ai.output.messages": [
                    {"role": "assistant", "parts": [{"type": "text", "content": json.dumps(answers)}]}
                ],
            }
            if self._capture_span:
                self._span.set_attributes(content)
            if self._capture_event:
                events.emit(
                    event_name="gen_ai.client.inference.operation.details",
                    severity_number=SeverityNumber.INFO,
                    attributes={**self._attributes, **usage, **content},
                )


@contextmanager
def inference(provider: str, operation: str, model: str, server_url: str) -> Generator[Inference]:
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
    with tracer.start_as_current_span(
        f"{operation} {model}",
        kind=trace.SpanKind.CLIENT,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        observed = Inference(span, attributes)
        started = time.perf_counter()
        try:
            yield observed
        except Exception as error:
            attributes["error.type"] = type(error).__qualname__
            span.set_attribute("error.type", attributes["error.type"])
            span.set_status(trace.StatusCode.ERROR)
            raise
        finally:
            inference_duration.record(time.perf_counter() - started, attributes)
