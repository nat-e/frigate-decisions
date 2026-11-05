from frigate_decisions.config import Rule
from frigate_decisions.rules import PADDING_S, End, RuleState, Start

RULE = Rule(name="dog_on_sofa", question="Is a dog on the sofa?", threshold=0.6, min_duration=5, end_after=10)


def feed(state: RuleState, answers: list[tuple[float, float]]) -> list:
    """Feed (time, probability) answers, checking for an end after each. Returns what happened."""
    happened = []
    for now, p in answers:
        if start := state.answer(p, now):
            happened.append(start)
            state.event_id = "evt"
        if end := state.check_end(now):
            happened.append(end)
    return happened


def test_starts_after_min_duration_of_yes():
    assert feed(RuleState(RULE), [(0, 0.9), (2, 0.8), (4, 0.7)]) == []
    assert feed(RuleState(RULE), [(0, 0.9), (2, 0.8), (4, 0.7), (6, 0.9)]) == [Start(0.9, 6 + PADDING_S)]


def test_a_no_before_the_start_resets_the_wait():
    # the flicker that made fake visits: yes, no, yes... never 5 s in a row
    answers = [(0, 0.61), (2, 0.4), (4, 0.62), (6, 0.5), (8, 0.61), (10, 0.3)]
    assert feed(RuleState(RULE), answers) == []


def test_a_no_during_the_event_does_not_end_it():
    happened = feed(RuleState(RULE), [(0, 0.9), (6, 0.9), (8, 0.2), (10, 0.9), (18, 0.2)])
    assert happened == [Start(0.9, 6 + PADDING_S)]


def test_ends_after_end_after_without_yes():
    state = RuleState(RULE)
    feed(state, [(0, 0.9), (6, 0.9), (8, 0.2)])
    assert state.check_end(15.9) is None
    assert state.check_end(16) == End("evt", 6 + PADDING_S)
    assert not state.busy


def test_pending_rule_gives_up_quietly_when_answers_stop():
    state = RuleState(RULE)
    feed(state, [(0, 0.9)])
    assert state.busy
    assert state.check_end(10) is None  # no event was created, nothing to end
    assert not state.busy


def test_zero_min_duration_starts_on_first_yes():
    rule = Rule(name="cat_seen", question="Is there a cat?", min_duration=0)
    assert feed(RuleState(rule), [(0, 0.3), (2, 0.7)]) == [Start(0.7, PADDING_S)]
