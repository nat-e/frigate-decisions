import io
from collections.abc import Iterator
from typing import override

import requests
from conftest import Telemetry
from opentelemetry.sdk.metrics.export import Histogram, Sum
from PIL import Image

from frigate_decisions.config import Camera, Rule
from frigate_decisions.watcher import Watcher


def jpeg(shade: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (400, 300), (shade, shade, shade)).save(out, "JPEG")
    return out.getvalue()


class FakeFrigate:
    def __init__(self, frames: list[bytes]):
        self.frames: Iterator[bytes] = iter(frames)
        self.created: list[tuple[str, str, str | None]] = []
        self.ended: list[str] = []

    def latest_jpeg(self, camera: str) -> bytes:
        return next(self.frames)

    def create_event(
        self, camera: str, label: str, sub_label: str | None, score: float, pre_capture: float
    ) -> str | None:
        self.created.append((camera, label, sub_label))
        return f"evt{len(self.created)}"

    def end_event(self, event_id: str, end_time: float) -> None:
        self.ended.append(event_id)

    def retry_ends(self) -> None:
        pass

    def end_leftover_events(self, camera: str, labels: set[str]) -> None:
        pass


class FakeProvider:
    def __init__(self, answers: list[float]):
        self.answers: Iterator[float] = iter(answers)
        self.calls: int = 0

    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        self.calls += 1
        return {name: next(self.answers) for name in questions}


CAMERA = Camera(
    name="living_room",
    rules=(Rule(name="dog_on_sofa", question="Dog on the sofa?", label="dog", sub_label="Sofa", min_duration=4),),
    ask_every=2,
    idle_after=20,
)


def run(frames: list[bytes], answers: list[float], seconds: int) -> tuple[FakeFrigate, FakeProvider]:
    frigate, provider = FakeFrigate(frames), FakeProvider(answers)
    watcher = Watcher(CAMERA, frigate, provider)
    for now in range(seconds):
        watcher.step(float(now))
    return frigate, provider


def test_no_movement_no_questions() -> None:
    _, provider = run([jpeg(100)] * 30, [], 30)
    assert provider.calls == 0


def test_movement_then_dog_creates_and_ends_one_event() -> None:
    frames = [jpeg(100), jpeg(200)] + [jpeg(200)] * 60  # one change at t=1, then still
    answers = [0.9, 0.9, 0.9] + [0.1] * 30  # dog for ~4 s, then gone
    frigate, provider = run(frames, answers, 60)
    assert frigate.created == [("living_room", "dog", "Sofa")]
    assert frigate.ended == ["evt1"]
    assert provider.calls < 20  # stopped asking once quiet


def test_decision_and_event_measurements(telemetry: Telemetry) -> None:
    frames = [jpeg(100), jpeg(200)] + [jpeg(200)] * 60
    _, provider = run(frames, [0.9, 0.9, 0.9] + [0.1] * 30, 60)
    collected = telemetry.collect()
    decisions = collected["frigate_decisions.decisions"].data
    assert isinstance(decisions, Sum)
    assert sum(point.value for point in decisions.data_points) == provider.calls
    yes = next(point for point in decisions.data_points if point.attributes and point.attributes["answer"] == "yes")
    assert yes.value == 3
    assert yes.attributes == {"camera": "living_room", "rule": "dog_on_sofa", "answer": "yes"}
    events = collected["frigate_decisions.rule.events"].data
    assert isinstance(events, Sum)
    assert {point.attributes["action"]: point.value for point in events.data_points if point.attributes} == {
        "started": 1,
        "end_requested": 1,
    }
    duration = collected["frigate_decisions.rule.event.duration"].data
    assert isinstance(duration, Histogram)
    assert duration.data_points[0].count == 1
    assert duration.data_points[0].sum == 4


def test_failed_event_create_does_not_count_as_started(telemetry: Telemetry) -> None:
    class FailingFrigate(FakeFrigate):
        @override
        def create_event(
            self,
            camera: str,
            label: str,
            sub_label: str | None,
            score: float,
            pre_capture: float,
        ) -> str | None:
            return None

    camera = Camera("test", (Rule("dog", "Dog?"),))
    watcher = Watcher(camera, FailingFrigate([]), FakeProvider([0.9]))
    watcher.ask(jpeg(100), 1)
    collected = telemetry.collect()
    if "frigate_decisions.rule.events" in collected:
        data = collected["frigate_decisions.rule.events"].data
        assert isinstance(data, Sum)
        assert all(point.value == 0 for point in data.data_points)


def test_frame_failure_still_ends_rule_and_reports_error(telemetry: Telemetry) -> None:
    class FailingFrigate(FakeFrigate):
        @override
        def latest_jpeg(self, camera: str) -> bytes:
            raise requests.Timeout("no frame")

    camera = Camera("test", (Rule("dog", "Dog?", end_after=1),))
    frigate = FailingFrigate([])
    watcher = Watcher(camera, frigate, FakeProvider([0.9]))
    watcher.ask(jpeg(100), 1)
    watcher.step(2)
    assert frigate.ended == ["evt1"]
    data = telemetry.collect()["frigate_decisions.frames"].data
    assert isinstance(data, Sum)
    point = next(point for point in data.data_points if point.value)
    assert point.value == 1
    assert point.attributes == {"camera": "test", "outcome": "error", "error.type": "Timeout"}


def test_stop_wakes_an_idle_watcher() -> None:
    watcher = Watcher(CAMERA, FakeFrigate([]), FakeProvider([]))
    watcher.start()
    watcher.stop()
    watcher.join(timeout=2)
    assert not watcher.is_alive()
