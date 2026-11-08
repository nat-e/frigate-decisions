"""Entry point: python -m frigate_decisions --config config.yml"""

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import cast

from . import __doc__ as description
from .config import Config, ConfigError, load
from .frigate import Frigate
from .logging import configure
from .providers import Provider, make_provider
from .watcher import Watcher

# Leaves time to flush telemetry within Docker's 10 s stop grace period
STOP_TIMEOUT_S = 5


def main() -> None:
    parser = argparse.ArgumentParser(prog="frigate-decisions", description=description)
    _ = parser.add_argument("--config", type=Path, default=Path("/config/config.yml"))
    _ = parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=os.environ.get("FRIGATE_DECISIONS_LOG_FORMAT", "text"),
        help="log output format (default: FRIGATE_DECISIONS_LOG_FORMAT, or text)",
    )
    args = parser.parse_args()
    config_path = cast("Path", args.config)
    log_format = cast("str", args.log_format)
    if log_format not in ("text", "json"):
        parser.error(f"invalid FRIGATE_DECISIONS_LOG_FORMAT: {log_format!r} (choose text or json)")
    configure(log_format)

    try:
        config = load(config_path)
        provider = make_provider(config.provider.name, config.provider.model)
        check_question_limit(config, provider)
    except (ConfigError, ValueError) as e:
        sys.exit(f"frigate-decisions: {e}")

    watchers: list[Watcher] = []
    for camera in config.cameras:
        frigate = Frigate(config.frigate.url)
        frigate.end_leftover_events(camera.name, {rule.label for rule in camera.rules})
        watchers.append(Watcher(camera, frigate, provider))

    stopping = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopping.set()
        # A second signal stops at once
        _ = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        _ = signal.signal(signal.SIGINT, signal.default_int_handler)

    _ = signal.signal(signal.SIGTERM, stop)
    _ = signal.signal(signal.SIGINT, stop)
    for watcher in watchers:
        watcher.start()
    _ = stopping.wait()
    for watcher in watchers:
        watcher.stop()
    # Watchers in a model call may not stop in time; they are daemon threads and end with the process.
    deadline = time.monotonic() + STOP_TIMEOUT_S
    for watcher in watchers:
        watcher.join(max(0.0, deadline - time.monotonic()))


def check_question_limit(config: Config, provider: Provider) -> None:
    """Every camera's rules are asked in one request, so they must fit the provider's limit."""
    limit = provider.max_questions
    for camera in config.cameras:
        if limit is not None and len(camera.rules) > limit:
            raise ConfigError(f"cameras.{camera.name}: {config.provider.name} takes at most {limit} rules per camera")


if __name__ == "__main__":
    main()
