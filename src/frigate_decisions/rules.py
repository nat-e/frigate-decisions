"""Turns a rule's stream of answers into events: when one starts, and when it ends."""

from dataclasses import dataclass

from . import telemetry
from .config import Rule

PADDING_S = 5  # extra video kept before and after each event


@dataclass(frozen=True)
class Start:
    score: float
    start_time: float  # Unix time the event begins


@dataclass(frozen=True)
class End:
    event_id: str
    end_time: float
    duration: float  # first-to-last yes, without video padding or the end delay


class RuleState:
    """One rule on one camera.

    The event starts once the answer has been "yes" for `min_duration` seconds without a "no" in
    between, and ends once there has been no "yes" for `end_after` seconds. A "no" during an event
    doesn't end it: a cat can turn around in the box. Records the rule's answer and event metrics.
    """

    def __init__(self, rule: Rule, camera: str):
        self.rule: Rule = rule
        self._first_yes: float | None = None
        self._last_yes: float = 0.0
        self._event_id: str | None = None
        self._attributes: dict[str, str] = {"camera": camera, "rule": rule.name}
        telemetry.rule_active.add(0, self._attributes)

    @property
    def busy(self) -> bool:
        """True while the rule is on, or on its way: keep asking."""
        return self._first_yes is not None

    def answer(self, probability: float, now: float) -> Start | None:
        """Record one answer. Returns a Start when the event should begin.

        Until `started()` is called, each later "yes" returns a Start again, so creating the event
        is retried, and a "no" still resets the wait.
        """
        yes = probability >= self.rule.threshold
        telemetry.probability.record(probability, self._attributes)
        telemetry.decisions.add(1, {**self._attributes, "answer": "yes" if yes else "no"})
        if yes:
            if self._first_yes is None:
                self._first_yes = now
            self._last_yes = now
            if self._event_id is None and now - self._first_yes >= self.rule.min_duration:
                return Start(score=probability, start_time=self._first_yes - PADDING_S)
        elif self._event_id is None:
            self._first_yes = None  # interrupted before the event started: start over
        return None

    def started(self, event_id: str) -> None:
        """Call once the event of a Start exists in Frigate."""
        self._event_id = event_id
        telemetry.rule_active.add(1, self._attributes)
        telemetry.rule_events.add(1, {**self._attributes, "action": "started"})

    def check_end(self, now: float) -> End | None:
        """Call regularly. Returns an End when the event should end."""
        if self._first_yes is None or now - self._last_yes < self.rule.end_after:
            return None
        return self._end(self._last_yes + PADDING_S)

    def finish(self, now: float) -> End | None:
        """Call on shutdown. Returns an End for an event still open."""
        if self._first_yes is None:
            return None
        return self._end(min(now, self._last_yes + PADDING_S))

    def _end(self, end_time: float) -> End | None:
        assert self._first_yes is not None
        event_id, duration = self._event_id, max(0, self._last_yes - self._first_yes)
        self._first_yes, self._event_id = None, None
        if not event_id:
            return None
        telemetry.rule_active.add(-1, self._attributes)
        telemetry.rule_events.add(1, {**self._attributes, "action": "end_requested"})
        telemetry.event_duration.record(duration, self._attributes)
        return End(event_id, end_time, duration)
