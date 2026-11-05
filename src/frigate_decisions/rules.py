"""Turns a rule's stream of answers into events: when one starts, and when it ends."""

from dataclasses import dataclass

from .config import Rule

PADDING_S = 5  # extra video kept before and after each event


@dataclass(frozen=True)
class Start:
    score: float
    start_time: float


@dataclass(frozen=True)
class End:
    event_id: str
    end_time: float


class RuleState:
    """One rule on one camera.

    The event starts once the answer has been "yes" for `min_duration` seconds without a "no" in
    between, and ends once there has been no "yes" for `end_after` seconds. A "no" during an event
    doesn't end it: a cat can turn around in the box.
    """

    def __init__(self, rule: Rule):
        self.rule: Rule = rule
        self.first_yes: float | None = None
        self.last_yes: float = 0.0
        self._event_id: str | None = None

    @property
    def busy(self) -> bool:
        """True while the rule is on, or on its way: keep asking."""
        return self.first_yes is not None

    def answer(self, probability: float, now: float) -> Start | None:
        """Record one answer. Returns a Start when the event should begin.

        Until `started()` is called, each later "yes" returns a Start again, so creating the event
        is retried, and a "no" still resets the wait.
        """
        if probability >= self.rule.threshold:
            if self.first_yes is None:
                self.first_yes = now
            self.last_yes = now
            if self._event_id is None and now - self.first_yes >= self.rule.min_duration:
                return Start(score=probability, start_time=self.first_yes - PADDING_S)
        elif self._event_id is None:
            self.first_yes = None  # interrupted before the event started: start over
        return None

    def started(self, event_id: str) -> None:
        """Call once the event of a Start exists in Frigate."""
        self._event_id = event_id

    def check_end(self, now: float) -> End | None:
        """Call regularly. Returns an End when the event should end."""
        if self.first_yes is None or now - self.last_yes < self.rule.end_after:
            return None
        event_id, end_time = self._event_id, self.last_yes + PADDING_S
        self.first_yes, self._event_id = None, None
        return End(event_id, end_time) if event_id else None
