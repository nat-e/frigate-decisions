"""Watches one camera: asks the questions while something moves, and creates the Frigate events."""

import io
import threading
import time
from typing import Protocol, override

import requests
import structlog
from PIL import Image, ImageChops, ImageFilter

from .config import Camera
from .providers import Provider
from .rules import RuleState

POLL_S = 1  # how often to grab a frame for the motion check

log = structlog.stdlib.get_logger(__name__)


class FrigateAPI(Protocol):
    """The Frigate calls a Watcher makes. frigate.Frigate makes them for real; tests fake them."""

    def latest_jpeg(self, camera: str) -> bytes: ...

    def create_event(
        self, camera: str, label: str, sub_label: str | None, score: float, pre_capture: float
    ) -> str | None: ...

    def end_event(self, event_id: str, end_time: float) -> None: ...

    def retry_ends(self) -> None: ...

    def end_leftover_events(self, camera: str, labels: set[str]) -> None: ...


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
        self.rules: list[RuleState] = [RuleState(rule) for rule in camera.rules]
        self.questions: dict[str, str] = {rule.name: rule.question for rule in camera.rules}
        self.motion: MotionCheck = MotionCheck()
        self.last_motion: float = float("-inf")
        self.last_ask: float = float("-inf")
        self.active: bool = False

    @override
    def run(self) -> None:
        self.frigate.end_leftover_events(self.camera.name, {rule.label for rule in self.camera.rules})
        self.log.info("watching", rule_count=len(self.rules))
        while True:
            time.sleep(POLL_S)
            try:
                self.step(time.time())
            except Exception:  # keep watching whatever happens
                self.log.exception("unexpected error")

    def step(self, now: float) -> None:
        cam = self.camera
        self.frigate.retry_ends()
        try:
            jpeg = self.frigate.latest_jpeg(cam.name)
        except requests.RequestException as e:
            self.log.warning("can't get a frame from Frigate", error=str(e))
            jpeg = None

        if jpeg is not None:
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
        try:
            answers = self.provider.ask(jpeg, self.camera.context, self.questions)
        except Exception as e:  # noqa: BLE001 any network or API error: skip this frame
            self.log.warning("decision model failed", error=str(e))
            return

        for state in self.rules:
            rule = state.rule
            rule_log = self.log.bind(rule=rule.name)
            probability = answers[rule.name]
            rule_log.info("decision", probability=probability)
            if start := state.answer(probability, now):
                state.event_id = self.frigate.create_event(
                    self.camera.name, rule.label, rule.sub_label, start.score, start.pre_capture
                )
                if state.event_id is not None:
                    rule_log.info("rule started", event_id=state.event_id, probability=start.score)
