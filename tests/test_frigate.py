import io
import json
import time
from typing import cast, override

import requests
from conftest import Telemetry
from opentelemetry.sdk.metrics.export import Sum
from requests.adapters import BaseAdapter

from frigate_decisions.frigate import Frigate

BODY = b'{"event_id": "evt"}'


class FakeServer(BaseAdapter):
    """Answers the client's requests in place of Frigate, each with the next reply: a status or an error."""

    def __init__(self, *replies: int | Exception):
        super().__init__()
        self.replies: list[int | Exception] = list(replies)
        self.requests: list[requests.PreparedRequest] = []

    @override
    def send(self, request: requests.PreparedRequest, *args: object, **kwargs: object) -> requests.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        response = requests.Response()
        response.status_code, response.raw = reply, io.BytesIO(BODY)
        return response

    @override
    def close(self) -> None:
        pass

    def sent(self) -> list[tuple[str | None, str]]:
        return [(r.method, r.path_url) for r in self.requests]


def client(server: FakeServer, url: str = "http://frigate:5000") -> Frigate:
    frigate = Frigate(url)
    frigate.http.mount("http://", server)
    return frigate


def test_a_frame_frigate_fails_to_send_is_none() -> None:
    frigate = client(FakeServer(requests.ConnectionError("refused"), 500, 200))
    assert frigate.latest_jpeg("door") is None
    assert frigate.latest_jpeg("door") is None
    assert frigate.latest_jpeg("door") == BODY


def test_an_end_frigate_refuses_is_retried_with_the_next_frame() -> None:
    server = FakeServer(503, 200, 200, 200)
    frigate = client(server)
    frigate.end_event("evt", 100.0)
    _ = frigate.latest_jpeg("door")
    _ = frigate.latest_jpeg("door")
    end, frame = ("PUT", "/api/events/evt/end"), ("GET", "/api/door/latest.jpg?quality=90")
    assert server.sent() == [end, end, frame, frame]


def test_a_trailing_slash_in_the_url_is_ignored() -> None:
    server = FakeServer(200)
    _ = client(server, "http://frigate:5000/").latest_jpeg("door")
    assert server.requests[0].url == "http://frigate:5000/api/door/latest.jpg?quality=90"


def test_the_event_start_is_sent_relative_to_now() -> None:
    server = FakeServer(200)
    assert client(server).create_event("door", "parcel", None, 0.9, start_time=time.time() - 12) == "evt"
    assert json.loads(cast("bytes", server.requests[0].body))["pre_capture"] == 12


def test_event_end_retry_keeps_pending_count_until_settled(telemetry: Telemetry) -> None:
    server = FakeServer(503, 200, 200, 404)
    frigate = client(server)
    frigate.end_event("first", 10)
    pending = telemetry.collect()["frigate_decisions.frigate.event.pending_ends"].data
    assert isinstance(pending, Sum)
    assert pending.data_points[0].value == 1
    _ = frigate.latest_jpeg("door")
    frigate.end_event("second", 20)
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
