import io
from collections.abc import Iterator, Sequence
from typing import override

import pytest
from conftest import Telemetry
from opentelemetry.sdk.metrics.export import Gauge, Histogram, Sum
from PIL import Image
from structlog.testing import capture_logs

from frigate_decisions.config import Camera, Rule
from frigate_decisions.watcher import Watcher, fit


def jpeg(shade: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (400, 300), (shade, shade, shade)).save(out, "JPEG")
    return out.getvalue()


class FakeFrigate:
    def __init__(self, frames: list[bytes]):
        self.frames: Iterator[bytes] = iter(frames)
        self.created: list[tuple[str, str, str | None]] = []
        self.ended: list[str] = []
        self.retries: int = 0

    def latest_jpeg(self, camera: str) -> bytes | None:
        return next(self.frames)

    def create_event(
        self, camera: str, label: str, sub_label: str | None, score: float, start_time: float
    ) -> str | None:
        self.created.append((camera, label, sub_label))
        return f"evt{len(self.created)}"

    def end_event(self, event_id: str, end_time: float) -> None:
        self.ended.append(event_id)

    def retry_ends(self) -> None:
        self.retries += 1


class FakeProvider:
    image_sides: tuple[int, int] = (280, 1568)
    max_questions: int | None = None

    def __init__(self, answers: Sequence[float | None]):
        self.answers: Iterator[float | None] = iter(answers)
        self.calls: int = 0
        self.images: list[bytes] = []
        self.questions: dict[str, str] = {}

    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        self.calls += 1
        self.images.append(image)
        self.questions = questions
        answers = {name: next(self.answers) for name in questions}
        return {name: p for name, p in answers.items() if p is not None}  # None: the model refused


CAMERA = Camera(
    name="living_room",
    rules=(Rule(name="dog_on_sofa", question="Dog on the sofa?", label="dog", sub_label="Sofa", min_duration=4),),
    ask_every=2,
    idle_after=20,
)


def run(frames: list[bytes], answers: Sequence[float | None], seconds: int) -> tuple[FakeFrigate, FakeProvider]:
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
            start_time: float,
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
        def latest_jpeg(self, camera: str) -> bytes | None:
            return None

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
    assert point.attributes == {"camera": "test", "outcome": "error"}


def test_stop_wakes_an_idle_watcher() -> None:
    watcher = Watcher(CAMERA, FakeFrigate([]), FakeProvider([]))
    watcher.start()
    watcher.stop()
    watcher.join(timeout=2)
    assert not watcher.is_alive()


def test_stop_ends_the_open_event() -> None:
    frigate = FakeFrigate([])
    watcher = Watcher(CAMERA, frigate, FakeProvider([0.9, 0.9, 0.9]))
    for now in (0, 2, 4):
        watcher.ask(jpeg(100), now)
    assert frigate.created
    watcher.start()
    watcher.stop()
    watcher.join(timeout=2)
    assert frigate.ended == ["evt1"]


def test_stop_without_open_events_retries_queued_ends() -> None:
    frigate = FakeFrigate([])
    watcher = Watcher(CAMERA, frigate, FakeProvider([]))
    watcher.start()
    watcher.stop()
    watcher.join(timeout=2)
    assert (frigate.ended, frigate.retries) == ([], 1)


def test_failed_step_still_reports_camera_activity(telemetry: Telemetry) -> None:
    frigate = FakeFrigate([jpeg(100), b"not a JPEG"])
    watcher = Watcher(CAMERA, frigate, FakeProvider([]))
    watcher.step(0)
    watcher.active = True
    with pytest.raises(OSError):
        watcher.step(1)
    data = telemetry.collect()["frigate_decisions.camera.active"].data
    assert isinstance(data, Gauge)
    point = next(point for point in data.data_points if point.attributes == {"camera": "living_room"})
    assert point.value == 1


def test_refusal_is_counted_without_fabricating_a_probability(telemetry: Telemetry) -> None:
    watcher = Watcher(CAMERA, FakeFrigate([]), FakeProvider([None]))
    watcher.ask(jpeg(100), 1)
    collected = telemetry.collect()
    decisions = collected["frigate_decisions.decisions"].data
    assert isinstance(decisions, Sum)
    point = next(point for point in decisions.data_points if point.value)
    assert point.attributes is not None
    assert point.attributes["answer"] == "unanswered"
    if "frigate_decisions.decision.probability" in collected:
        probabilities = collected["frigate_decisions.decision.probability"].data
        assert isinstance(probabilities, Histogram)
        assert all(point.count == 0 for point in probabilities.data_points)


def test_provider_gets_resized_image_and_framed_question() -> None:
    big = io.BytesIO()
    Image.new("RGB", (3840, 2160), (200, 200, 200)).save(big, "JPEG")
    frames = [jpeg(100), big.getvalue()] + [jpeg(200)] * 5
    _, provider = run(frames, [0.1] * 5, 5)
    assert Image.open(io.BytesIO(provider.images[0])).size == (1568, 882)
    assert provider.questions == {"dog_on_sofa": "Looking at the camera image: Dog on the sofa?"}


def test_refused_answer_skips_only_that_frame() -> None:
    frames = [jpeg(100), jpeg(200)] + [jpeg(200)] * 60
    answers: list[float | None] = [0.9, None, 0.9, 0.9, 0.9] + [0.1] * 30
    frigate, _ = run(frames, answers, 60)
    assert frigate.created == [("living_room", "dog", "Sofa")]


def test_fit_keeps_images_already_in_range() -> None:
    image = jpeg(100)
    assert fit(image, 280, 1568) == image


def test_fit_scales_small_images_up() -> None:
    assert Image.open(io.BytesIO(fit(jpeg(100), 400, 1568))).size == (533, 400)


def test_one_decision_per_rule_with_separate_camera_context() -> None:
    rules = (Rule(name="cat", question="Cat?"), Rule(name="dog", question="Dog?"))
    with capture_logs() as logs:
        for name, answers in (("litter", [0.91, 0.12]), ("door", [0.23, 0.84])):
            watcher = Watcher(Camera(name=name, rules=rules), FakeFrigate([]), FakeProvider(answers))
            watcher.ask(jpeg(100), 0)

    decisions = [event for event in logs if event["event"] == "decision"]
    assert [(event["camera"], event["rule"], event["probability"]) for event in decisions] == [
        ("litter", "cat", 0.91),
        ("litter", "dog", 0.12),
        ("door", "cat", 0.23),
        ("door", "dog", 0.84),
    ]
    starts = [event for event in logs if event["event"] == "rule started"]
    assert [(event["camera"], event["rule"], event["event_id"]) for event in starts] == [
        ("litter", "cat", "evt1"),
        ("door", "dog", "evt1"),
    ]


def test_end_log_keeps_event_id_after_rule_resets() -> None:
    with capture_logs() as logs:
        _ = run([jpeg(100), jpeg(200)] + [jpeg(200)] * 60, [0.9, 0.9, 0.9] + [0.1] * 30, 60)
    ends = [event for event in logs if event["event"] == "rule ended"]
    assert len(ends) == 1
    assert ends[0]["camera"] == "living_room"
    assert ends[0]["rule"] == "dog_on_sofa"
    assert ends[0]["event_id"] == "evt1"
