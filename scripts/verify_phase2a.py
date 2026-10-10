#!/usr/bin/env python3
"""Run the complete Phase 2A matrix or one configured scientific experiment.

Outputs are immutable new directories. --all runs every declared configuration,
including the full 100-seed CPU Monte Carlo experiments; pytest keeps only fast
strict checks. No training or Git write operation occurs here.
"""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import os
import platform
import subprocess
import sys
import time
import tomllib
import traceback
import numpy as np
import scipy
from fatigue_dic_rl.evaluation.identifiability_verification import clean
from fatigue_dic_rl.evaluation.sequential_verification import RUNNERS
from fatigue_dic_rl.evaluation.belief_plots import plot_phase2a


def metadata(root, config_path, config):
    hashes = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
              for base in (root/"src", root/"scripts") for p in sorted(base.rglob("*.py"))}
    source_sha = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "git_head": head,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "source_sha256": source_sha, "source_hashes": hashes,
        "code_version": head+"+worktree-source-sha256:"+source_sha,
        "config": config, "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "command": [sys.executable, *sys.argv],
        "thread_environment": {n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "noise_contract": "frame noise independent conditional on shared physical/registration parameters; full within-frame covariance",
        "units": {"coordinates": "mm", "displacement": "mm", "covariance": "mm^2", "damage": "1", "A": "cycle^-1"},
        "assumptions": "fixed Tri3 mesh; local scalar damage; plane stress; proportionate fatigue; local Gaussian parameter approximation"}


def run_one(root, config_path, output):
    config = tomllib.loads(config_path.read_text())
    case = config["case"]
    if case not in RUNNERS:
        raise ValueError("unknown Phase 2A case: "+case)
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.toml").write_bytes(config_path.read_bytes())
    provenance = metadata(root, config_path, config)
    begin = time.perf_counter()
    try:
        result = RUNNERS[case](config, output)
        figures = plot_phase2a(case, result, output)
        result["figures"] = figures
    except Exception as exc:
        result = {"status": "FAILED", "exception_type": type(exc).__name__, "message": str(exc),
                  "traceback": traceback.format_exc()}
    result.update(metadata=provenance, elapsed_seconds=time.perf_counter()-begin)
    (output/"summary.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False)+"\n")
    rows = clean(result.get("rows", []))
    if rows:
        with (output/"metrics.csv").open("w", newline="") as stream:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    manifest = {str(p.relative_to(output)): sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob("*")) if p.is_file()}
    (output/"artifact_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"case": case, "status": result["status"], "seconds": result["elapsed_seconds"], "output": str(output)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.all:
        args.output.mkdir(parents=True, exist_ok=False)
        configs = {p.stem: p for p in (root/"configs"/"phase2a").glob("*.toml")}
        if set(configs) != set(RUNNERS):
            raise RuntimeError("scientific matrix and declared configurations differ")
        statuses = []
        for case in RUNNERS:
            result = run_one(root, configs[case], args.output/case)
            statuses.append({"case": case, "status": result["status"], "seconds": result["elapsed_seconds"],
                             "source_sha256": result["metadata"]["source_sha256"]})
        passed = all(s["status"] == "PASSED" for s in statuses)
        (args.output/"matrix_summary.json").write_text(json.dumps({"status": "PASSED" if passed else "FAILED", "cases": statuses}, indent=2)+"\n")
    else:
        passed = run_one(root, args.config, args.output)["status"] == "PASSED"
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
