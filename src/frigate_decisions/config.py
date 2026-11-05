"""The YAML config: which cameras to watch, and the questions to ask about them."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


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
        data = cast("object", yaml.safe_load(path.read_text()))  # any YAML value: parse() checks it
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"can't read {path}: {e}") from e
    return parse(data)


def parse(data: object) -> Config:
    root = _mapping(data, "config")
    _check_keys(root, {"frigate", "provider", "cameras"}, "config")
    frigate = _mapping(root.get("frigate"), "frigate")
    _check_keys(frigate, {"url"}, "frigate")
    provider = _mapping(root.get("provider"), "provider")
    _check_keys(provider, {"name", "model"}, "provider")
    cameras_data = _mapping(root.get("cameras"), "cameras")
    if not cameras_data:
        raise ConfigError("config: add at least one camera under 'cameras'")

    cameras: list[Camera] = []
    for cam_name, cam_data in cameras_data.items():
        where = f"cameras.{cam_name}"
        cam = _mapping(cam_data, where)
        rules_data = _mapping(cam.pop("rules", None), f"{where}.rules")
        if not rules_data:
            raise ConfigError(f"{where}: add at least one rule under 'rules'")
        rules = tuple(
            _build(Rule, {"label": name, **_mapping(rule, f"{where}.rules.{name}")}, f"{where}.rules.{name}", name=name)
            for name, rule in rules_data.items()
        )
        cameras.append(_build(Camera, cam, where, name=cam_name, rules=rules))

    model = provider.get("model")
    return Config(
        frigate_url=str(frigate.get("url", "http://frigate:5000")).rstrip("/"),
        provider=str(provider.get("name", "jev")),
        model=None if model is None else str(model),
        cameras=tuple(cameras),
    )


def _mapping(value: object, where: str) -> dict[str, object]:
    """`value` as a mapping with text keys: an empty one when the YAML key has no value."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping of keys to values, got {value!r}")
    return {str(k): v for k, v in cast("dict[object, object]", value).items()}


def _build[T: DataclassInstance](cls: type[T], data: dict[str, object], where: str, **fixed: object) -> T:
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


def _check_keys(data: dict[str, object], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}, expected some of {sorted(allowed)}")
