from pathlib import Path

import pytest

from frigate_decisions.config import ConfigError, load, parse

EXAMPLE = Path(__file__).parent.parent / "examples" / "config.yml"


def test_example_config_loads():
    config = load(EXAMPLE)
    assert config.provider == "jev"
    assert config.cameras[0].rules[0].question


def test_defaults_and_label():
    config = parse({"cameras": {"door": {"rules": {"parcel": {"question": "Is there a parcel?"}}}}})
    assert config.frigate_url == "http://frigate:5000"
    rule = config.cameras[0].rules[0]
    assert (rule.name, rule.label, rule.threshold, rule.min_duration, rule.end_after) == ("parcel", "parcel", 0.6, 0, 10)


@pytest.mark.parametrize(
    "data, message",
    [
        ({}, "at least one camera"),
        ({"cameras": {"door": {}}}, "at least one rule"),
        ({"cameras": {"door": {"rules": {"parcel": {}}}}}, "question"),
        ({"cameras": {"door": {"rules": {"parcel": {"question": "?", "for": 5}}}}}, "unknown key"),
        ({"cameras": {"door": {"rules": {"parcel": {"question": "?", "min_duration": "5s"}}}}}, "expected a number"),
    ],
)
def test_clear_errors(data, message):
    with pytest.raises(ConfigError, match=message):
        parse(data)
