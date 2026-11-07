from collections.abc import Generator
from dataclasses import dataclass

import pytest
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader, Metric
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@dataclass
class Telemetry:
    spans: InMemorySpanExporter
    logs: InMemoryLogRecordExporter
    reader: InMemoryMetricReader

    def collect(self) -> dict[str, Metric]:
        data = self.reader.get_metrics_data()
        assert data is not None
        return {
            metric.name: metric
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }


@pytest.fixture(scope="session")
def telemetry_sdk() -> Generator[Telemetry]:
    spans = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    trace.set_tracer_provider(tracer_provider)
    reader = InMemoryMetricReader(
        preferred_temporality={Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    )
    meter_provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)
    logs = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
    set_logger_provider(logger_provider)
    yield Telemetry(spans, logs, reader)
    logger_provider.shutdown()
    meter_provider.shutdown()
    tracer_provider.shutdown()


@pytest.fixture
def telemetry(telemetry_sdk: Telemetry) -> Telemetry:
    telemetry_sdk.spans.clear()
    telemetry_sdk.logs.clear()
    _ = telemetry_sdk.reader.get_metrics_data()
    return telemetry_sdk
