"""Watches one camera: asks the questions while something moves, and creates the Frigate events."""

import io
import threading
import time
from typing import Protocol, override

import structlog
from opentelemetry.trace import StatusCode
from PIL import Image, ImageChops, ImageFilter

from . import telemetry
from .config import Camera
from .providers import Provider
from .rules import RuleState

POLL_S = 1  # how often to grab a frame for the motion check
QUESTION_PREFIX = "Looking at the camera image: "  # every provider gets the same wording

log = structlog.stdlib.get_logger(__name__)


class FrigateAPI(Protocol):
    """The Frigate calls a Watcher makes. frigate.Frigate makes them for real; tests fake them."""

    def latest_jpeg(self, camera: str) -> bytes | None: ...

    def create_event(
        self, camera: str, label: str, sub_label: str | None, score: float, start_time: float
    ) -> str | None: ...

    def end_event(self, event_id: str, end_time: float) -> None: ...

    def retry_ends(self) -> None: ...


class MotionCheck:
    """Cheap frame-to-frame difference on a small blurred grayscale copy."""

    SIZE: tuple[int, int] = (200, 150)

    def __init__(self):
        self.prev: Image.Image | None = None

    def changed(self, jpeg: bytes) -> float:
        """Share of pixels that changed since the previous frame."""
        # Pillow's type for `size` includes numpy arrays, unknown without numpy installed
        small = Image.open(io.BytesIO(jpeg)).convert("L").resize(self.SIZE)  # pyright: ignore[reportUnknownMemberType]
        cur = small.filter(ImageFilter.GaussianBlur(2))
        prev, self.prev = self.prev, cur
        if prev is None:
            return 0.0
        histogram = ImageChops.difference(cur, prev).histogram()
        return sum(histogram[26:]) / (self.SIZE[0] * self.SIZE[1])  # pixels that changed by more than 25


class Watcher(threading.Thread):
    def __init__(self, camera: Camera, frigate: FrigateAPI, provider: Provider):
        super().__init__(name=camera.name, daemon=True)
        self.camera: Camera = camera
        self.log: structlog.stdlib.BoundLogger = log.bind(camera=camera.name)
        self.frigate: FrigateAPI = frigate
        self.provider: Provider = provider
        self.rules: list[RuleState] = [RuleState(rule, camera.name) for rule in camera.rules]
        self.questions: dict[str, str] = {rule.name: QUESTION_PREFIX + rule.question for rule in camera.rules}
        self.motion: MotionCheck = MotionCheck()
        self.last_motion: float = float("-inf")
        self.last_ask: float = float("-inf")
        self.active: bool = False
        self._stopping: threading.Event = threading.Event()

    def stop(self) -> None:
        """Finish the current poll, end the open events and stop."""
        self._stopping.set()

    @override
    def run(self) -> None:
        self.log.info("watching", rule_count=len(self.rules))
        while not self._stopping.wait(POLL_S):
            try:
                self.step(time.time())
            except Exception as error:  # keep watching whatever happens
                telemetry.watcher_errors.add(1, {"camera": self.camera.name, "error.type": type(error).__qualname__})
                self.log.exception("unexpected error")
        try:
            self.finish(time.time())
        except Exception:
            self.log.exception("can't end the open events")

    def finish(self, now: float) -> None:
        """End the open events, and retry the ends Frigate hasn't accepted yet."""
        ends = [(state.rule.name, end) for state in self.rules if (end := state.finish(now))]
        for rule_name, end in ends:
            self.log.info("ending event on shutdown", rule=rule_name, event_id=end.event_id)
            self.frigate.end_event(end.event_id, end.end_time)  # also retries the earlier ends
        if not ends:
            self.frigate.retry_ends()

    def step(self, now: float) -> None:
        try:
            self._step(now)
        finally:
            telemetry.camera_active.set(int(self.active), {"camera": self.camera.name})

    def _step(self, now: float) -> None:
        cam = self.camera
        jpeg = self.frigate.latest_jpeg(cam.name)
        telemetry.frames.add(1, {"camera": cam.name, "outcome": "success" if jpeg is not None else "error"})
        if jpeg is not None:
            telemetry.last_frame.set(now, {"camera": cam.name})
            moved = self.motion.changed(jpeg)
            if moved >= cam.motion_threshold:
                self.last_motion = now
                if not self.active:
                    self.log.info("movement, asking", motion_fraction=moved)

            was_active = self.active
            self.active = now - self.last_motion < cam.idle_after or any(r.busy for r in self.rules)
            if was_active and not self.active:
                self.log.info("quiet again")
            if self.active and now - self.last_ask >= cam.ask_every:
                self.last_ask = now
                self.ask(jpeg, now)

        for state in self.rules:
            if end := state.check_end(now):
                self.log.info("rule ended", rule=state.rule.name, event_id=end.event_id)
                self.frigate.end_event(end.event_id, end.end_time)

    def ask(self, jpeg: bytes, now: float) -> None:
        with telemetry.tracer.start_as_current_span(
            "frigate_decisions.decide",
            attributes={"camera": self.camera.name},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                image = fit(jpeg, *self.provider.image_sides)
                answers = self.provider.ask(image, self.camera.context, self.questions)
            except Exception as e:  # noqa: BLE001 any network or API error: skip this frame
                span.set_attribute("error.type", type(e).__qualname__)
                span.set_status(StatusCode.ERROR)
                self.log.warning("decision model failed", error=str(e))
                return

            for state in self.rules:
                rule = state.rule
                rule_log = self.log.bind(rule=rule.name)
                if rule.name not in answers:
                    telemetry.decisions.add(1, {"camera": self.camera.name, "rule": rule.name, "answer": "unanswered"})
                    rule_log.warning("no answer")
                    continue
                probability = answers[rule.name]
                rule_log.info("decision", probability=probability)
                if not (start := state.answer(probability, now)):
                    continue
                event_id = self.frigate.create_event(
                    self.camera.name, rule.label, rule.sub_label, start.score, start.start_time
                )
                if event_id:  # otherwise the next "yes" tries again
                    state.started(event_id)
                    rule_log.info("rule started", event_id=event_id, probability=start.score)


def fit(jpeg: bytes, min_side: int, max_side: int) -> bytes:
    """Scale the JPEG so its sides are between `min_side` and `max_side` pixels, keeping its shape."""
    img = Image.open(io.BytesIO(jpeg))
    scale = min(1.0, max_side / max(img.size))
    scale = max(scale, min_side / min(img.size))
    if scale == 1.0:
        return jpeg
    out = io.BytesIO()
    size = (round(img.width * scale), round(img.height * scale))
    # Pillow's type for `size` includes numpy arrays, unknown without numpy installed
    resized = img.convert("RGB").resize(size)  # pyright: ignore[reportUnknownMemberType]
    resized.save(out, "JPEG", quality=90)
    return out.getvalue()
