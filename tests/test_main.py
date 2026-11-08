import pytest

from frigate_decisions.__main__ import check_question_limit
from frigate_decisions.config import Camera, Config, ConfigError, FrigateSettings, ProviderSettings, Rule


class LimitedProvider:
    image_sides: tuple[int, int] = (280, 1568)
    max_questions: int | None = 2

    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        return {}


def config(rule_count: int) -> Config:
    rules = tuple(Rule(name=f"r{i}", question="?") for i in range(rule_count))
    return Config(FrigateSettings(), ProviderSettings(name="openai"), cameras=(Camera("door", rules),))


def test_rules_within_the_provider_limit_pass() -> None:
    check_question_limit(config(2), LimitedProvider())


def test_too_many_rules_for_the_provider_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"cameras.door: openai takes at most 2 rules"):
        check_question_limit(config(3), LimitedProvider())
