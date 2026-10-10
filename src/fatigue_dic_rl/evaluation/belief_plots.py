"""Static exportable scientific figures; labels preserve temporal semantics."""
from pathlib import Path
import numpy as np


def plot_phase2a(case, result, folder):
    if result.get("status") != "PASSED":
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    files = []
    if case == "baseline_comparison":
        fig, ax = plt.subplots(figsize=(8, 4.5))
        rows = result["rows"]
        for name in dict.fromkeys(r["baseline"] for r in rows):
            rs = [r for r in rows if r["baseline"] == name]
            current = [r for r in rs if "current_damage_rmse" in r]
            future = [r for r in rs if "damage_rmse" in r]
            line = ax.plot([r["cycle_count"] for r in current], [r["current_damage_rmse"] for r in current], "o-", label=name)[0]
            if future:
                ax.plot([r["cycle_count"] for r in future], [r["damage_rmse"] for r in future], "^--", color=line.get_color())
        ax.axvline(20., color="black", linestyle=":", label="assimilation cutoff N=20")
        ax.set(xlabel="Cycle count N", ylabel="Damage RMSE", title="Filtered states (circles) and frozen-prefix forecasts (triangles)")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
        files.append(("baseline_current_and_forecast", fig))
    elif case in {"coverage_uniform", "coverage_fatigue"}:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        rows = result["rows"]
        names = [r["scenario"] for r in rows]
        for name in dict.fromkeys(n for r in rows for n in r["parameters"]):
            values = [r["parameters"].get(name, {}).get("coverage", {}).get("operational_coverage_including_failed_or_unreliable", np.nan) for r in rows]
            axes[0].plot(names, values, "o-", label=name)
        for i, cycle in enumerate((30, 40)):
            axes[1].plot(names, [r["future"][i]["damage_coverage"]["operational_coverage_including_failed_or_unreliable"] for r in rows], "o-", label=f"forecast N={cycle}")
        for ax in axes:
            ax.axhline(.95, linestyle=":", color="black", label="nominal 95%")
            ax.set_ylim(-.03, 1.03); ax.tick_params(axis="x", rotation=40); ax.set_ylabel("Coverage over independent seeds")
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        axes[0].set_title("Parameter intervals (failed/unreliable included)")
        axes[1].set_title("Open-loop damage intervals from N<=20")
        files.append(("coverage", fig))
    elif case == "model_discrepancy":
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        names = [r["case"] for r in result["rows"]]
        axes[0].bar(names, [r["prefix_whitened_rmse"] for r in result["rows"]])
        axes[0].set(ylabel="Measurement-whitened residual RMS", title="Prefix residuals retain model discrepancy")
        for row in result["rows"]:
            axes[1].plot([r["cycle_count"] for r in row["future"]], [r["damage_rmse"] for r in row["future"]], "o-", label=row["case"])
        axes[1].set(xlabel="Future cycle N (cutoff 20)", ylabel="Damage forecast RMSE", title="No future assimilation")
        axes[0].tick_params(axis="x", rotation=25); axes[1].legend(fontsize=8)
        files.append(("model_discrepancy", fig))
    output = []
    for name, fig in files:
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = folder/(name+"."+suffix)
            if path.exists(): raise FileExistsError(path)
            fig.savefig(path, dpi=170)
            output.append(path.name)
        plt.close(fig)
    return output
