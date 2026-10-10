"""Standalone figures from saved raw fields; flat element maps, no damage smoothing."""

from pathlib import Path

import numpy as np


def plot_fatigue(case: str, data: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(folder/f"{name}.{suffix}")
        plt.close(fig)

    if case == "temporal_convergence":
        rows = data["rows"]
        dt = np.array([r["delta_N_cycle"] for r in rows])
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
        keys = ["damage_rms_error", "u_rms_error_mm", "stress_tensor_rms_error_n_mm2",
                "strain_amplitude_tensor_rms_error", "elastic_energy_abs_error_n_mm", "cumulative_dissipation_abs_error_n_mm"]
        labels = ["Damage RMS error (1)", "Displacement RMS error (mm)", "Stress RMS error (N/mm²)",
                  "Strain amplitude RMS error (1)", "Elastic energy error (N·mm)", "Damage dissipation estimate error (N·mm)"]
        for ax, key, ylabel in zip(axes.flat, keys, labels, strict=True):
            values = np.array([r[key] for r in rows])
            ax.loglog(dt, values, "o-", color="#245a81", label="FE staggered")
            ax.loglog(dt, values[0]*dt/dt[0], "--", color="#555555", label="First-order reference")
            ax.set(xlabel="Cycle increment ΔN (cycle)", ylabel=ylabel)
            ax.grid(True, color="#dddddd", linewidth=.6)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle("Fixed mesh fatigue time convergence | independent DOP853 reference")
        save(fig, "temporal_convergence")
    elif case == "uniform_damage":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, control in zip(axes, ("load", "displacement"), strict=True):
            with np.load(folder/f"{control}_1/truth/trajectory.npz") as d:
                ax.plot(d["N"], d["damage"].mean(axis=1), "o-", color="#245a81", label="FE fatigue")
            row = next(r for r in data["rows"] if r["control"] == control)
            ax.scatter([row["N_cycle"]], [row["analytic_final_damage"]], facecolors="none", edgecolors="#555555",
                       marker="s", s=60, label="Independent analytic final D")
            ax.set(xlabel="N (cycle)", ylabel="Uniform damage D (1)", title=f"{control.capitalize()} control", ylim=(0, .25))
            ax.grid(True, color="#dddddd", linewidth=.6)
            ax.legend(fontsize=9)
        save(fig, "uniform_damage_evolution")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, control, key, label in zip(axes, ("load", "displacement"),
                ("max_displacement_mm", "max_reaction_n"), ("Right displacement (mm)", "Right reaction (N)"), strict=True):
            with np.load(folder/f"{control}_1/truth/trajectory.npz") as d:
                nodes = d["nodes_mm"]
                ids = np.flatnonzero(nodes[:, 0] == nodes[:, 0].max())
                values = d[key][:, ids, 0].mean(axis=1) if control == "load" else d[key][:, ids, 0].sum(axis=1)
                ax.plot(d["N"], values, "o-", color="#245a81")
            ax.set(xlabel="N (cycle)", ylabel=label, title=f"{control.capitalize()} control")
            ax.grid(True, color="#dddddd", linewidth=.6)
        save(fig, "control_response")
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for control, color, style in (("load", "#245a81", "o-"), ("displacement", "#a96518", "s--")):
            with np.load(folder/f"{control}_1/truth/trajectory.npz") as d:
                ax.plot(d["N"], d["max_elastic_energy_n_mm"], style, color=color, label=f"{control} control")
        ax.set(xlabel="N (cycle)", ylabel="Maximum-phase elastic energy (N·mm)")
        ax.grid(True, color="#dddddd", linewidth=.6)
        ax.legend()
        save(fig, "control_energy")
    elif case == "mesh_sensitivity":
        trajectories = []
        for row in data["rows"]:
            with np.load(folder/f"mesh_{row['nx']}/truth/trajectory.npz") as d:
                trajectories.append({k: d[k] for k in ("nodes_mm", "connectivity", "damage")})
        vmin = min(d["damage"][-1].min() for d in trajectories)
        vmax = max(d["damage"][-1].max() for d in trajectories)
        fig, axes = plt.subplots(1, len(trajectories), figsize=(12, 3.8), layout="constrained")
        for ax, d, row in zip(axes, trajectories, data["rows"], strict=True):
            tri = mtri.Triangulation(d["nodes_mm"][:, 0], d["nodes_mm"][:, 1], d["connectivity"])
            im = ax.tripcolor(tri, facecolors=d["damage"][-1], vmin=vmin, vmax=vmax, cmap="cividis", shading="flat")
            ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"{row['E']} Tri3 cells")
            ax.set_aspect("equal")
        fig.colorbar(im, ax=axes, label="Final local damage D (1)", shrink=.75)
        fig.suptitle("Mesh sensitivity | flat local fields, common color scale")
        save(fig, "mesh_damage_fields")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for history, color, marker in zip(data["histories"], ("#245a81", "#a96518", "#68733e"), ("o", "s", "^"), strict=True):
            rows = history["rows"]
            for ax, key, ylabel in zip(axes, ("damage_max", "right_u_mm"), ("Maximum D (1)", "Right mean u (mm)"), strict=True):
                ax.plot([r["N_cycle"] for r in rows], [r[key] for r in rows], color=color,
                        marker=marker, markevery=max(1, len(rows)//8), label=f"nx={history['nx']}")
                ax.set(xlabel="N (cycle)", ylabel=ylabel)
                ax.grid(True, color="#dddddd", linewidth=.6)
        axes[0].legend()
        save(fig, "mesh_evolution")
    if case in {"localized_damage", "synthetic_trajectory"}:
        with np.load(folder/"trajectory/truth/trajectory.npz") as raw:
            d = {k: raw[k] for k in raw.files}
        tri = mtri.Triangulation(d["nodes_mm"][:, 0], d["nodes_mm"][:, 1], d["connectivity"])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        for i, (ax, values, title) in enumerate(zip(axes, (d["damage"][0], d["damage"][-1], d["damage"][-1]-d["damage"][0]),
                ("Initial imperfection D", "Final evolved D", "Evolved increment ΔD"), strict=True)):
            im = ax.tripcolor(tri, facecolors=values, shading="flat", cmap="cividis", vmin=0,
                             vmax=d["damage"].max() if i < 2 else (d["damage"][-1]-d["damage"][0]).max())
            ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, label="1", shrink=.7)
        fig.suptitle("Local scalar damage | separate ΔD scale; no spatial smoothing")
        save(fig, "damage_fields")
        amplitude = np.sqrt(np.sum(d["strain_amplitude"]**2*[1, 1, .5], axis=2))
        for name, values, unit in (("strain_amplitude_fields", amplitude, "Strain tensor amplitude norm (1)"),
                                   ("stress_redistribution", d["max_stress_n_mm2"][:, :, 0], "Maximum-phase σxx (N/mm²)")):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
            for ax, i, title in zip(axes, (0, -1), ("Initial", "Final"), strict=True):
                im = ax.tripcolor(tri, facecolors=values[i], shading="flat", cmap="viridis",
                                 vmin=values.min(), vmax=values.max())
                ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
                ax.set_aspect("equal")
            fig.colorbar(im, ax=axes, label=unit, shrink=.75)
            save(fig, name)
    if case in {"constant_hazard", "reversed_loading", "localized_damage", "synthetic_trajectory"}:
        rows = data["rows"]
        N = [r["N_cycle"] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for key, color, style, label in (("damage_mean", "#245a81", "o-", "Area mean"),
                                        ("damage_max", "#a96518", "s--", "Maximum")):
            axes[0].plot(N, [r[key] for r in rows], style, color=color, label=label)
        axes[0].set(ylabel="Damage D (1)")
        axes[0].legend(fontsize=9)
        axes[1].plot(N, [r["right_u_mm"] for r in rows], "o-", color="#245a81")
        axes[1].set(ylabel="Right mean u (mm)")
        axes[2].plot(N, [r["elastic_energy_n_mm"] for r in rows], "o-", color="#245a81", label="Elastic U")
        axes[2].plot(N, [r["cumulative_max_phase_damage_dissipation_estimate_n_mm"] for r in rows], "s--",
                     color="#a96518", label="Σ Ymax ΔD estimate")
        axes[2].set(ylabel="Energy (N·mm)")
        axes[2].legend(fontsize=9)
        for ax in axes:
            ax.set(xlabel="N (cycle)")
            ax.grid(True, color="#dddddd", linewidth=.6)
        fig.suptitle(f"{case.replace('_', ' ').capitalize()} | physics-only synthetic")
        save(fig, "histories")
