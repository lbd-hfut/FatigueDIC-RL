#!/usr/bin/env python3
"""Run with installed package or PYTHONPATH=src; optionally load the example TOML."""

import argparse
import json
from pathlib import Path
import tomllib

from fatigue_dic_rl.smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    settings = {}
    if args.config:
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)
        settings = dict(config["damage"])
        settings["cycle_steps"] = tuple(settings["cycle_steps"])
        observation = config["observation"]
        if observation != {"coordinate_units": "mm", "coordinate_system": "specimen"}:
            parser.error("this analytic fixture uses mm coordinates in the specimen frame")
    print(json.dumps(run_smoke(**settings), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
