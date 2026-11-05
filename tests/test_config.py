from pathlib import Path

import pytest

from frigate_decisions.config import ConfigError, load, parse

EXAMPLE = Path(__file__).parent.parent / "examples" / "config.yml"


def test_example_config_loads() -> None:
    config = load(EXAMPLE)
    assert config.provider.name == "jev"
    assert config.cameras[0].rules[0].question


def test_defaults_and_label() -> None:
    config = parse({"cameras": {"door": {"rules": {"parcel": {"question": "Is there a parcel?"}}}}})
    assert (config.frigate.url, config.provider.name, config.provider.model) == ("http://frigate:5000", "jev", None)
    rule = config.cameras[0].rules[0]
    assert (rule.name, rule.label, rule.threshold, rule.min_duration, rule.end_after) == (
        "parcel",
        "parcel",
        0.6,
        0,
        10,
    )


def test_keys_without_value_get_their_default() -> None:
    config = parse({"cameras": {"door": {"rules": {"parcel": {"question": "?", "label": None, "threshold": None}}}}})
    rule = config.cameras[0].rules[0]
    assert (rule.label, rule.threshold) == ("parcel", 0.6)


@pytest.mark.parametrize(
    "data, message",
    [
        (None, "at least one camera"),
        ({}, "at least one camera"),
        ({"cameras": ["door"]}, "cameras: expected a mapping"),
        ({"cameras": {"door": {"rules": ["parcel"]}}}, "cameras.door.rules: expected a mapping"),
        ({"cameras": {"door": {}}}, "at least one rule"),
        ({"cameras": {"door": {"rules": {"parcel": {}}}}}, "cameras.door.rules.parcel: missing 'question'"),
        ({"cameras": {"door": {"rules": {"parcel": {"question": True}}}}}, "question: expected text"),
        ({"cameras": {"door": {"rules": {"parcel": {"question": "?", "for": 5}}}}}, "unknown key"),
        ({"cameras": {"door": {"rules": {"parcel": {"question": "?", "min_duration": "5s"}}}}}, "expected a number"),
    ],
)
def test_clear_errors(data: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        _ = parse(data)
