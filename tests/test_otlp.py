"""Exercise the documented launcher against a local OTLP/HTTP receiver."""

import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import override

import pytest


@pytest.mark.parametrize(("launch", "disabled"), [(True, False), (True, True), (False, False)])
def test_standard_launcher_exports_all_signals(
    tmp_path: Path,
    launch: bool,
    disabled: bool,
) -> None:
    launcher = shutil.which("opentelemetry-instrument")
    if launcher is None:
        pytest.skip("install the otel extra to test the launcher")
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    received: dict[str, list[bytes]] = {}

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.setdefault(self.path, []).append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()

        @override
        def log_message(self, format: str, *args: object) -> None:
            pass

    script = tmp_path / "emit.py"
    _ = script.write_text("""
import logging
from frigate_decisions.telemetry import inference

logging.basicConfig(level=logging.INFO)
with inference("typesafe", "system_one", "jev-test", "https://api.example.com") as observed:
    observed.response("jev-resolved", 100, 4, None, {"dog": "Dog?"}, {"dog": 0.9})
    logging.getLogger("frigate_decisions").info("export smoke test")
""")
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
    with ThreadingHTTPServer(("127.0.0.1", 0), Receiver) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        env.update(
            {
                "OTEL_SERVICE_NAME": "frigate-decisions-test",
                "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT": "SPAN_AND_EVENT",
                "OTEL_SDK_DISABLED": str(disabled).lower(),
            }
        )
        command = [sys.executable, str(script)]
        if launch:
            command.insert(0, launcher)
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20, check=False)
            assert result.returncode == 0, result.stderr
        finally:
            server.shutdown()
            worker.join()

    if disabled or not launch:
        assert not received
        return

    traces = ExportTraceServiceRequest.FromString(received["/v1/traces"][0])
    resource = {attr.key: attr.value.string_value for attr in traces.resource_spans[0].resource.attributes}
    assert resource["service.name"] == "frigate-decisions-test"
    spans = [span for resource in traces.resource_spans for scope in resource.scope_spans for span in scope.spans]
    inference_span = next(span for span in spans if span.name == "system_one jev-test")
    attrs = {attr.key: attr.value for attr in inference_span.attributes}
    assert attrs["gen_ai.usage.input_tokens"].int_value == 100
    assert attrs["gen_ai.input.messages"].array_value.values[0].kvlist_value.values

    metrics = [ExportMetricsServiceRequest.FromString(payload) for payload in received["/v1/metrics"]]
    names = {
        metric.name
        for request in metrics
        for resource in request.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert "gen_ai.client.inference.duration" in names
    assert "gen_ai.client.inference.usage.input_tokens" in names

    logs = [ExportLogsServiceRequest.FromString(payload) for payload in received["/v1/logs"]]
    records = [
        record
        for request in logs
        for resource in request.resource_logs
        for scope in resource.scope_logs
        for record in scope.log_records
    ]
    assert any(record.body.string_value == "export smoke test" for record in records)
    assert any(record.event_name == "gen_ai.client.inference.operation.details" for record in records)
