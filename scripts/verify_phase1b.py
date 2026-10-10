#!/usr/bin/env python3
"""Run one Phase 1B scientific experiment into a new, reproducible output directory."""

import argparse
import csv
from hashlib import sha256
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

from fatigue_dic_rl.evaluation.fatigue_plots import plot_fatigue
from fatigue_dic_rl.evaluation.fatigue_verification import RUNNERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if config.get("case") not in RUNNERS:
        parser.error("unknown Phase 1B verification case")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"config.toml").write_bytes(args.config.read_bytes())
    file_handler = logging.FileHandler(args.output/"solver.log")
    file_handler.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", handlers=[file_handler, stream])
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
              for base in (root/"src", root/"scripts") for p in sorted(base.rglob("*.py"))}
    source_sha = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    config["code_version"] = f"{head}+worktree-source-sha256:{source_sha}"
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "command": [sys.executable, *sys.argv], "config": config,
        "config_sha256": sha256(args.config.read_bytes()).hexdigest(), "source_hashes": hashes,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "git_head": head, "code_version": config["code_version"]}
    try:
        result = RUNNERS[config["case"]](config, args.output)
        plot_fatigue(config["case"], result, args.output)
        rows = result.get("rows", [])
        if rows:
            with (args.output/"error_table.csv").open("w", newline="") as handle:
                keys = list(dict.fromkeys(k for row in rows for k in row))
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        output = dict(result, metadata=metadata)
    except Exception as exc:
        output = {"status": "FAILED", "exception_type": type(exc).__name__, "message": str(exc),
                  "traceback": traceback.format_exc(), "metadata": metadata}
        if hasattr(exc, "failures"):
            from dataclasses import asdict
            output["failure_diagnostics"] = [asdict(f) for f in exc.failures]
        (args.output/"summary.json").write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
        logging.exception("Phase 1B verification failed; diagnostics preserved")
        raise
    (args.output/"summary.json").write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"case": config["case"], "status": output["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
