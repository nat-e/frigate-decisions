import io
from collections.abc import Iterator

from PIL import Image
from structlog.testing import capture_logs

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


def test_one_decision_per_rule_with_separate_camera_context() -> None:
    rules = (Rule(name="cat", question="Cat?"), Rule(name="dog", question="Dog?"))
    with capture_logs() as logs:
        for name, answers in (("litter", [0.91, 0.12]), ("door", [0.23, 0.84])):
            watcher = Watcher(Camera(name=name, rules=rules), FakeFrigate([]), FakeProvider(answers))
            watcher.ask(b"frame", 0)

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
