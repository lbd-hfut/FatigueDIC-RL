#!/usr/bin/env python3
"""Run one reproducible scientific verification case and preserve all fields/errors."""

import argparse
import csv
from dataclasses import asdict
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
import traceback

import numpy as np
import scipy

from fatigue_dic_rl.evaluation.tri3_verification import RUNNERS
from fatigue_dic_rl.evaluation.tri3_plots import plot_verification
from fatigue_dic_rl.mechanics import MechanicsConvergenceError
from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if config.get("case") not in RUNNERS:
        parser.error("unknown verification case")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"config.toml").write_bytes(args.config.read_bytes())
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output/"solver.log")])
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "command": [sys.executable, *sys.argv], "config": config,
                "solver_configuration": asdict(Tri3SolverConfig(**config.get("solver", {}))),
                "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    try:
        result = RUNNERS[config["case"]](config, args.output)
        plot_verification(config["case"], result, args.output)
        rows = result.get("rows", result.get("uniform_rows", result.get("patch_rows")))
        if rows:
            keys = list(dict.fromkeys(k for row in rows for k in row))
            with (args.output/"error_table.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        output = dict(result, metadata=metadata)
    except Exception as exc:
        output = {"status": "FAILED", "exception": type(exc).__name__, "message": str(exc),
                  "traceback": traceback.format_exc(), "metadata": metadata}
        if isinstance(exc, MechanicsConvergenceError):
            output["diagnostics"] = asdict(exc.diagnostics)
        (args.output/"summary.json").write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
        logging.exception("Scientific verification failed; evidence preserved")
        raise
    (args.output/"summary.json").write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
