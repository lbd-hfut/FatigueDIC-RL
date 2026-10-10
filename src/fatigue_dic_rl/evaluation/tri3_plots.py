"""Standalone scientific figures; each plotted point comes from saved verification rows."""

from pathlib import Path

import numpy as np


def plot_verification(case: str, data: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    if case == "convergence":
        rows = data["rows"]
        h = np.array([r["h_max_mm"] for r in rows])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
        for ax, key, order, color, title in zip(axes,
                ["relative_displacement_L2", "relative_strain_L2"], [2, 1], ["#245a81", "#a96518"],
                ["Displacement L2 error", "Strain tensor L2 error"]):
            error = np.array([r[key] for r in rows])
            fit = float(np.polyfit(np.log(h), np.log(error), 1)[0])
            ax.loglog(h, error, "o-", color=color, linewidth=1.6, label=f"Computed (fit p={fit:.3f})")
            ax.loglog(h, error[0]*(h/h[0])**order, "--", color="#555555", linewidth=1.2, label=f"O(h^{order}) reference")
            ax.set(xlabel="Maximum element edge h (mm)", ylabel="Relative L2 error (dimensionless)", title=title)
            ax.set_xticks(h)
            ax.set_xticklabels([f"{value:.3f}" for value in h])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
            ax.legend(fontsize=9)
        fig.suptitle("Tri3 manufactured-solution convergence | unstructured meshes")
        for suffix in ["png", "svg"]:
            fig.savefig(folder/f"convergence.{suffix}")
        plt.close(fig)
    elif case == "damage":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
        for ax, control, key, label in zip(axes, ["load", "displacement"],
                ["right_mean_u_mm", "right_reaction_n"], ["Displacement / undamaged displacement", "Reaction / undamaged reaction"]):
            rows = [r for r in data["uniform_rows"] if r["control"] == control]
            damage = np.array([r["damage"] for r in rows])
            baseline = next(r[key] for r in rows if r["damage"] == 0)
            ratio = np.array([r[key]/baseline for r in rows])
            ax.plot(damage, ratio, "o", color="#245a81", label="Computed FEM")
            samples = np.linspace(0, damage.max(), 101)
            analytic = 1/(1-samples) if control == "load" else 1-samples
            ax.plot(samples, analytic, "--", color="#555555", linewidth=1.2, label="Analytical scaling")
            ax.set(xlabel="Fixed uniform damage D", ylabel=label, title=f"{control.capitalize()} control", ylim=(0, max(1.1, ratio.max()*1.1)))
            ax.grid(True, color="#dddddd", linewidth=0.6)
            ax.legend(fontsize=9)
        fig.suptitle("Fixed stiffness degradation | no fatigue evolution")
        for suffix in ["png", "svg"]:
            fig.savefig(folder/f"damage_control.{suffix}")
        plt.close(fig)
