"""Entry point: python -m frigate_decisions --config config.yml"""

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

from . import __doc__ as description
from .config import ConfigError, load
from .frigate import Frigate
from .providers import make_provider
from .watcher import Watcher


def main() -> None:
    parser = argparse.ArgumentParser(prog="frigate-decisions", description=description)
    _ = parser.add_argument("--config", type=Path, default=Path("/config/config.yml"))
    config_path = cast("Path", parser.parse_args().config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx2", "httpcore2", "urllib3", "typesafe_sdk"):  # one line per HTTP request otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        config = load(config_path)
        provider = make_provider(config.provider.name, config.provider.model)
    except (ConfigError, ValueError) as e:
        sys.exit(f"frigate-decisions: {e}")

    watchers = [Watcher(camera, Frigate(config.frigate.url), provider) for camera in config.cameras]
    for watcher in watchers:
        watcher.start()
    for watcher in watchers:
        watcher.join()


if __name__ == "__main__":
    main()
