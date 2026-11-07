import io
from collections.abc import Iterator, Sequence

from PIL import Image

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

    def latest_jpeg(self, camera: str) -> bytes:
        return next(self.frames)

    def create_event(
        self, camera: str, label: str, sub_label: str | None, score: float, start_time: float
    ) -> str | None:
        self.created.append((camera, label, sub_label))
        return f"evt{len(self.created)}"

    def end_event(self, event_id: str, end_time: float) -> None:
        self.ended.append(event_id)


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
