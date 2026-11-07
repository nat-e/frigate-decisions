"""The few Frigate API calls we need.

Uses Frigate's internal API (port 5000), which has no login: run this service on the same Docker
network as Frigate and don't expose that port.
"""

import logging
import time
from typing import NotRequired, TypedDict, cast

import requests

from . import telemetry

log = logging.getLogger(__name__)


class _CreatedEvent(TypedDict):
    """Frigate's reply to POST /api/events/<camera>/<label>/create"""

    event_id: str


class _Event(TypedDict):
    """One event in Frigate's reply to GET /api/events, with the fields we read"""

    id: str
    label: str
    start_time: float
    data: NotRequired[dict[str, object] | None]


class Frigate:
    def __init__(self, url: str):
        self.url: str = url
        self.http: requests.Session = requests.Session()
        self.pending_ends: list[tuple[str, float]] = []  # retried until Frigate accepts them

    def latest_jpeg(self, camera: str) -> bytes:
        r = self.http.get(f"{self.url}/api/{camera}/latest.jpg", params={"quality": 90}, timeout=10)
        r.raise_for_status()
        return r.content

    def create_event(
        self, camera: str, label: str, sub_label: str | None, score: float, pre_capture: float
    ) -> str | None:
        """Open an event that stays open until end_event(). Returns its id, or None on failure."""
        body = {
            "sub_label": sub_label,
            "score": round(score, 2),
            "duration": None,
            "include_recording": True,
            "pre_capture": round(pre_capture),
        }
        try:
            r = self.http.post(f"{self.url}/api/events/{camera}/{label}/create", json=body, timeout=10)
            r.raise_for_status()
            event_id = cast("_CreatedEvent", r.json())["event_id"]
        except (requests.RequestException, KeyError, ValueError) as e:
            telemetry.event_operations.add(
                1, {"operation": "create", "outcome": "error", "error.type": type(e).__qualname__}
            )
            log.warning("%s: can't create a %s event in Frigate: %s", camera, label, e)
            return None
        telemetry.event_operations.add(1, {"operation": "create", "outcome": "success"})
        return event_id

    def end_event(self, event_id: str, end_time: float) -> None:
        self.pending_ends.append((event_id, end_time))
        telemetry.pending_ends.add(1)
        self.retry_ends()

    def retry_ends(self) -> None:
        pending, self.pending_ends = self.pending_ends, []
        for event_id, end_time in pending:
            try:
                r = self.http.put(f"{self.url}/api/events/{event_id}/end", json={"end_time": end_time}, timeout=10)
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
            except requests.RequestException as e:
                telemetry.event_operations.add(
                    1, {"operation": "end", "outcome": "error", "error.type": type(e).__qualname__}
                )
                log.warning("can't end event %s yet: %s", event_id, e)
                self.pending_ends.append((event_id, end_time))
            else:
                telemetry.pending_ends.add(-1)
                telemetry.event_operations.add(
                    1, {"operation": "end", "outcome": "discarded" if r.status_code >= 400 else "success"}
                )

    def end_leftover_events(self, camera: str, labels: set[str]) -> None:
        """Events left open by a previous run would keep their review item open forever: end them."""
        try:
            r = self.http.get(
                f"{self.url}/api/events",
                params={"cameras": camera, "labels": ",".join(sorted(labels)), "in_progress": 1, "limit": 100},
                timeout=10,
            )
            r.raise_for_status()
            for event in cast("list[_Event]", r.json()):
                if (event.get("data") or {}).get("type") == "api":
                    log.info(
                        "%s: ending event %s (%s) left open by a previous run", camera, event["id"], event["label"]
                    )
                    self.end_event(event["id"], min(time.time(), event["start_time"] + 60))
        except (requests.RequestException, ValueError) as e:
            log.warning("%s: can't check for events left open: %s", camera, e)
