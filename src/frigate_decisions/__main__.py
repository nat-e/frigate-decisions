"""Entry point: python -m frigate_decisions --config config.yml"""

import argparse
import os
import sys
from pathlib import Path
from typing import cast

from . import __doc__ as description
from .config import ConfigError, load
from .frigate import Frigate
from .logging import configure
from .providers import make_provider
from .watcher import Watcher


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
        provider = make_provider(config.provider, config.model)
    except (ConfigError, ValueError) as e:
        sys.exit(f"frigate-decisions: {e}")

    watchers = [Watcher(camera, Frigate(config.frigate_url), provider) for camera in config.cameras]
    for watcher in watchers:
        watcher.start()
    for watcher in watchers:
        watcher.join()


if __name__ == "__main__":
    main()
