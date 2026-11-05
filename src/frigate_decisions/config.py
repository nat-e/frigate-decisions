"""The YAML config: which cameras to watch, and the questions to ask about them."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Rule:
    """A yes/no question about a camera image, and when its answer becomes a Frigate event."""

    name: str
    question: str
    threshold: float = 0.6  # "yes" when the model's probability is at least this
    min_duration: float = 0  # seconds of uninterrupted "yes" before the event starts
    end_after: float = 10  # seconds without a "yes" before the event ends
    label: str = ""  # Frigate label of the event, the rule name by default
    sub_label: str | None = None


@dataclass(frozen=True)
class Camera:
    name: str
    rules: tuple[Rule, ...]
    context: str | None = None  # what the camera shows, to help the model
    ask_every: float = 2  # seconds between questions while something is going on
    motion_threshold: float = 0.003  # share of pixels that must change to count as movement
    idle_after: float = 20  # seconds without movement (and no rule on) before questions stop


@dataclass(frozen=True)
class Config:
    frigate_url: str
    provider: str
    model: str | None
    cameras: tuple[Camera, ...]


def load(path: Path) -> Config:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"can't read {path}: {e}") from e
    return parse(data)


def parse(data: dict[str, Any]) -> Config:
    _check_keys(data, {"frigate", "provider", "cameras"}, "config")
    frigate = data.get("frigate") or {}
    _check_keys(frigate, {"url"}, "frigate")
    provider = data.get("provider") or {}
    _check_keys(provider, {"name", "model"}, "provider")
    if not data.get("cameras"):
        raise ConfigError("config: add at least one camera under 'cameras'")

    cameras = []
    for cam_name, cam in data["cameras"].items():
        where = f"cameras.{cam_name}"
        cam = dict(cam or {})
        rules = cam.pop("rules", None)
        if not rules:
            raise ConfigError(f"{where}: add at least one rule under 'rules'")
        parsed_rules = tuple(
            _build(Rule, {"label": name, **(rule or {})}, f"{where}.rules.{name}", name=name)
            for name, rule in rules.items()
        )
        cameras.append(_build(Camera, cam, where, name=cam_name, rules=parsed_rules))

    return Config(
        frigate_url=str(frigate.get("url", "http://frigate:5000")).rstrip("/"),
        provider=provider.get("name", "jev"),
        model=provider.get("model"),
        cameras=tuple(cameras),
    )


def _build(cls: type, data: dict[str, Any], where: str, **fixed: Any) -> Any:
    """Make a `cls` from `data`, with clear errors for unknown or missing keys."""
    allowed = {f.name for f in fields(cls)} - set(fixed)
    _check_keys(data, allowed, where)
    for f in fields(cls):
        value = data.get(f.name)
        if f.type is float and value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ConfigError(f"{where}.{f.name}: expected a number, got {value!r}")
    try:
        return cls(**data, **fixed)
    except TypeError as e:  # a required key is missing
        raise ConfigError(f"{where}: {e}") from e


def _check_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}, expected some of {sorted(allowed)}")
