from collections.abc import Iterator

import pytest
import requests
from conftest import Telemetry
from opentelemetry.sdk.metrics.export import Sum

from frigate_decisions.frigate import Frigate


def test_event_end_retry_keeps_pending_count_until_settled(
    telemetry: Telemetry, monkeypatch: pytest.MonkeyPatch
) -> None:
    frigate = Frigate("http://frigate")
    statuses: Iterator[int] = iter([503, 200, 404])

    def put(*args: object, **kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = next(statuses)
        return response

    monkeypatch.setattr(frigate.http, "put", put)
    frigate.end_event("first", 10)
    assert frigate.pending_ends == [("first", 10)]
    pending = telemetry.collect()["frigate_decisions.frigate.event.pending_ends"].data
    assert isinstance(pending, Sum)
    assert pending.data_points[0].value == 1
    frigate.retry_ends()
    frigate.end_event("second", 20)
    assert not frigate.pending_ends
    collected = telemetry.collect()
    pending = collected["frigate_decisions.frigate.event.pending_ends"].data
    assert isinstance(pending, Sum)
    assert pending.data_points[0].value == 0
    operations = collected["frigate_decisions.frigate.event.operations"].data
    assert isinstance(operations, Sum)
    assert {
        point.attributes["outcome"]: point.value for point in operations.data_points if point.attributes and point.value
    } == {
        "success": 1,
        "discarded": 1,
    }
