#!/usr/bin/env python3
"""Execute one reproducible Phase 1C experiment into a new output directory."""
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
from fatigue_dic_rl.evaluation.identifiability_verification import RUNNERS, clean
from fatigue_dic_rl.evaluation.identifiability_plots import plot_phase1c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if config.get("case") not in RUNNERS:
        parser.error("unknown Phase 1C experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"config.toml").write_bytes(args.config.read_bytes())
    logging.basicConfig(filename=args.output/"solver.log", level=logging.INFO)
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
              for base in (root/"src", root/"scripts") for p in sorted(base.rglob("*.py"))}
    source_sha = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "source_sha256": source_sha, "source_hashes": hashes, "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
        "config": config, "command": [sys.executable, *sys.argv],
        "assumptions": "2D small strain, isotropic plane stress, frozen local scalar damage; synthetic noise not experimentally calibrated",
        "units": {"coordinates": "mm", "displacement": "mm", "covariance": "mm^2", "stress": "N/mm^2", "reaction": "N"}}
    try:
        result = RUNNERS[config["case"]](config, args.output)
        output = dict(result, metadata=metadata)
        plot_phase1c(config["case"], output, args.output)
        rows = clean(result.get("rows", []))
        if rows:
            with (args.output/"error_table.csv").open("w", newline="") as f:
                keys = list(dict.fromkeys(k for row in rows for k in row))
                writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    except Exception as exc:
        output = {"status": "FAILED", "message": str(exc), "exception_type": type(exc).__name__,
                  "traceback": traceback.format_exc(), "metadata": metadata}
        (args.output/"summary.json").write_text(json.dumps(clean(output), indent=2, allow_nan=False)+"\n")
        logging.exception("Phase 1C failure; diagnostics retained")
        raise
    (args.output/"summary.json").write_text(json.dumps(clean(output), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"case": config["case"], "status": output["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
