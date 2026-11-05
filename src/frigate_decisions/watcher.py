"""Watches one camera: asks the questions while something moves, and creates the Frigate events."""

import io
import logging
import threading
import time

import requests
from PIL import Image, ImageChops, ImageFilter

from .config import Camera
from .frigate import Frigate
from .providers import Provider
from .rules import RuleState

POLL_S = 1  # how often to grab a frame for the motion check

log = logging.getLogger(__name__)


class MotionCheck:
    """Cheap frame-to-frame difference on a small blurred grayscale copy."""

    SIZE = (200, 150)

    def __init__(self):
        self.prev: Image.Image | None = None

    def changed(self, jpeg: bytes) -> float:
        """Share of pixels that changed since the previous frame."""
        cur = Image.open(io.BytesIO(jpeg)).convert("L").resize(self.SIZE).filter(ImageFilter.GaussianBlur(2))
        prev, self.prev = self.prev, cur
        if prev is None:
            return 0.0
        histogram = ImageChops.difference(cur, prev).histogram()
        return sum(histogram[26:]) / (self.SIZE[0] * self.SIZE[1])  # pixels that changed by more than 25


class Watcher(threading.Thread):
    def __init__(self, camera: Camera, frigate: Frigate, provider: Provider):
        super().__init__(name=camera.name, daemon=True)
        self.camera, self.frigate, self.provider = camera, frigate, provider
        self.rules = [RuleState(rule) for rule in camera.rules]
        self.questions = {rule.name: rule.question for rule in camera.rules}
        self.motion = MotionCheck()
        self.last_motion = self.last_ask = float("-inf")
        self.active = False

    def run(self) -> None:
        self.frigate.end_leftover_events(self.camera.name, {rule.label for rule in self.camera.rules})
        log.info("%s: watching, %d rule(s)", self.camera.name, len(self.rules))
        while True:
            time.sleep(POLL_S)
            try:
                self.step(time.time())
            except Exception:  # keep watching whatever happens
                log.exception("%s: unexpected error", self.camera.name)

    def step(self, now: float) -> None:
        cam = self.camera
        self.frigate.retry_ends()
        try:
            jpeg = self.frigate.latest_jpeg(cam.name)
        except requests.RequestException as e:
            log.warning("%s: can't get a frame from Frigate: %s", cam.name, e)
            jpeg = None

        if jpeg is not None:
            moved = self.motion.changed(jpeg)
            if moved >= cam.motion_threshold:
                self.last_motion = now
                if not self.active:
                    log.info("%s: movement (%.1f%% of pixels), asking", cam.name, moved * 100)

            was_active = self.active
            self.active = now - self.last_motion < cam.idle_after or any(r.busy for r in self.rules)
            if was_active and not self.active:
                log.info("%s: quiet again", cam.name)
            if self.active and now - self.last_ask >= cam.ask_every:
                self.last_ask = now
                self.ask(jpeg, now)

        for state in self.rules:
            if end := state.check_end(now):
                log.info("%s: %s ended", cam.name, state.rule.name)
                self.frigate.end_event(end.event_id, end.end_time)

    def ask(self, jpeg: bytes, now: float) -> None:
        try:
            answers = self.provider.ask(jpeg, self.camera.context, self.questions)
        except Exception as e:  # network or API error: skip this frame
            log.warning("%s: decision model failed: %s", self.camera.name, e)
            return
        log.info("%s: %s", self.camera.name, " ".join(f"{k}={v:.2f}" for k, v in answers.items()))

        for state in self.rules:
            rule = state.rule
            if start := state.answer(answers[rule.name], now):
                state.event_id = self.frigate.create_event(
                    self.camera.name, rule.label, rule.sub_label, start.score, start.pre_capture
                )
                log.info("%s: %s started: event %s", self.camera.name, rule.name, state.event_id)
