"""Standalone Phase 1C figures; all plotted values also exist in NPZ/JSON/CSV."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def finish(fig, folder, name):
    fig.tight_layout()
    fig.savefig(Path(folder)/(name+".png"), dpi=170)
    fig.savefig(Path(folder)/(name+".svg"))
    plt.close(fig)


def plot_phase1c(case, result, folder):
    folder = Path(folder)
    if case == "noise_covariance":
        for mode in ("independent", "spatial"):
            a = np.load(folder/f"noise_{mode}_0.03.npz")
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            vmax = max(a["covariance_mm2"].max(), a["empirical_covariance_mm2"].max())
            for ax, key, title in zip(axes, ("covariance_mm2", "empirical_covariance_mm2"), ("Specified covariance", "Repeated-sample covariance")):
                im = ax.imshow(a[key], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
                ax.set(title=f"{mode}: {title}", xlabel="Displacement component", ylabel="Displacement component")
                fig.colorbar(im, ax=ax, label="mm²", shrink=.7)
            finish(fig, folder, f"covariance_{mode}")
    if case == "sampling_density":
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
        for i, n in enumerate((16, 64, 256)):
            for j, sampling in enumerate(("regular", "random", "clustered")):
                seed = result["metadata"]["config"]["verification"]["seeds"][0] if "metadata" in result else 101
                a = np.load(folder/f"sampling_{n}_{sampling}_0_{seed}_map.npz")
                ax = axes[i, j]; ax.scatter(*a["coordinates_mm"].T, s=5)
                ax.set(xlim=(0, 2), ylim=(0, 1), title=f"{sampling}, Q={n}", xlabel="x (mm)", ylabel="y (mm)")
                ax.set_aspect("equal")
        finish(fig, folder, "observation_sampling_maps")
        fig, ax = plt.subplots(figsize=(8, 4))
        for sampling in ("regular", "random", "clustered"):
            vals = [[r["singular_values"][-1] for r in result["rows"] if r["count"] == n and r["sampling"] == sampling] for n in (16, 64, 256)]
            medians = np.median(vals, axis=1)
            ax.plot((16, 64, 256), medians, "o-", label=sampling)
            ax.fill_between((16, 64, 256), np.min(vals, axis=1), np.max(vals, axis=1), alpha=.14)
        ax.set(xscale="log", yscale="log", xlabel="Requested observation points", ylabel="Smallest whitened singular value",
               title="Sampling sensitivity: median and full seed/missing-mask range")
        ax.legend(); finish(fig, folder, "sampling_information")
    sensitivity_files = []
    if case == "gaussian_sensitivity":
        sensitivity_files = [folder/"gaussian_sensitivity.npz"]
        o = np.load(folder/"gaussian_observations"/"observation.npz")
        a = np.load(sensitivity_files[0]); j = a["jacobian_displacement_mm"].reshape(-1, 2, 6)
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
        for k, ax in enumerate(axes.ravel()):
            magnitude = np.linalg.norm(j[:, :, k], axis=1)
            im = ax.scatter(*o["coordinates_mm"].T, c=magnitude, cmap="viridis", s=24)
            ax.set(title=str(a["parameter_names"][k]), xlabel="x (mm)", ylabel="y (mm)", xlim=(0, 2), ylim=(0, 1))
            ax.set_aspect("equal"); fig.colorbar(im, ax=ax, label="mm / dimensionless p", shrink=.65)
        finish(fig, folder, "damage_parameter_sensitivity_maps")
    elif case == "multiple_loading":
        sensitivity_files = [folder/f"{n}.npz" for n in ("single", "same_mode_scaled", "independent_modes", "multiple_unknown_E")]
    elif case == "uniform_confounding":
        sensitivity_files = [folder/f"{n}.npz" for n in ("unknown_E_damage", "healthy_plus_damaged", "imposed_fisher_False_False", "imposed_fisher_True_True")]
    elif case == "multiframe":
        sensitivity_files = [folder/f"{n}.npz" for n in ("initial_single_frame", "current_single_frame_N20", "prefix_joint")]
    if sensitivity_files:
        fig, ax = plt.subplots(figsize=(8, 4))
        for path in sensitivity_files:
            a = np.load(path); s = a["singular_values"]
            ax.semilogy(np.arange(1, len(s)+1), np.maximum(s, 1e-12), "o-", label=path.stem)
        ax.set(xlabel="Singular direction", ylabel="Whitened Jacobian singular value",
               title="Local observability (values below 10⁻¹² displayed at 10⁻¹²)")
        ax.legend(fontsize=8); finish(fig, folder, "fisher_singular_spectra")
        a = np.load(sensitivity_files[0]); fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(a["parameter_correlation"], vmin=-1, vmax=1, cmap="RdBu_r")
        labels = a["parameter_names"]
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
        ax.set_title("Sensitivity column correlations")
        fig.colorbar(im, ax=ax); finish(fig, folder, "identifiability_correlation_heatmap")
    if case == "uniform_confounding":
        fig, ax = plt.subplots(figsize=(7, 4))
        young = np.linspace(1600., 3000., 200)
        ax.plot(young, 1-1600./young, label="Equivalent Eeff=1600 N/mm²")
        starts = result["multi_start_equivalent_solutions"]["starts"]
        ax.scatter([r["parameters"]["E0"] for r in starts], [r["parameters"]["d0"] for r in starts], label="All optimized starts")
        ax.scatter([2000.], [.2], marker="*", s=130, label="Synthetic truth (evaluation only)")
        ax.set(xlabel="E0 (N/mm²)", ylabel="Uniform damage D", title="Equal DIC fit does not select a unique E0 and D")
        ax.legend(); finish(fig, folder, "equivalent_parameter_solutions")
    if case == "inverse_recovery":
        rows = [r for r in result["rows"] if "damage_field_rmse" in r and r.get("loading") != "multiple_unknown_E"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for mode in ("single", "multiple"):
            rs = [r for r in rows if r["loading"] == mode and r.get("correlation_length_mm") is None]
            axes[0].scatter([r["noise_pixel"] for r in rs], [r["damage_field_rmse"] for r in rs], label=mode, alpha=.7)
            axes[1].scatter([r["displacement_rmse_mm"] for r in rs], [r["damage_field_rmse"] for r in rs], label=mode, alpha=.7)
        axes[0].set(xlabel="Noise standard deviation (pixel)", ylabel="Damage field RMSE")
        axes[1].set(xlabel="DIC displacement RMSE (mm)", ylabel="Damage field RMSE")
        for ax in axes: ax.legend(); ax.grid(alpha=.2)
        finish(fig, folder, "noise_recovery_comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, r in enumerate(rows):
            for start in r["starts"]:
                if start["cost"] is not None:
                    ax.scatter(i, max(start["cost"], 1e-16), marker="o" if start["converged"] else "x", color="C0" if start["converged"] else "C3", s=18)
        ax.set(yscale="log", xlabel="Prescribed recovery scenario (table row)", ylabel="Multi-start weighted objective",
               title="All initializations: circles converged, crosses did not converge")
        finish(fig, folder, "multistart_optimization_distribution")
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
        names = ("d_base", "amplitude", "xc", "yc", "sx", "sy")
        scales = (.075, .3, .7, .4, .29, .185)
        for k, ax in enumerate(axes.ravel()):
            for mode in ("single", "multiple"):
                rs = [r for r in rows if r["loading"] == mode and r.get("correlation_length_mm") is None]
                ax.scatter([r["noise_pixel"] for r in rs], [r["parameter_error"][names[k]]/scales[k] for r in rs], label=mode, s=18)
            ax.axhline(0, color=".4", lw=.6); ax.set(title=names[k], xlabel="Noise (pixel)", ylabel="Error / parameter scale")
        axes[0, 0].legend(fontsize=8); finish(fig, folder, "parameter_recovery_errors")
        # Illustrations use fixed scenario indices; every run remains in tables.
        field_files = [folder/"recovery_0_fields.npz", folder/"recovery_2_fields.npz"]
    elif case == "mesh_diagnostics":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        rows = result["rows"]
        axes[0].plot([r["quadrature_count"] for r in rows], [r["forward_observation_gap_mm"] for r in rows], "o-")
        axes[0].set(xlabel="Quadrature points", ylabel="Forward discrepancy (mm)")
        axes[1].plot([r["quadrature_count"] for r in rows], [r["cross_mesh_recovery"].get("damage_field_rmse", np.nan) for r in rows], "o-")
        axes[1].set(xlabel="Quadrature points", ylabel="Cross-mesh damage RMSE")
        finish(fig, folder, "mesh_sensitivity_comparison")
        field_files = [folder/f"mesh_{rows[0]['nx']}_cross_fit_fields.npz"]
    elif case == "boundary_uncertainty":
        field_files = [folder/"force_bias_absorbed_fields.npz", folder/"explicit_force_nuisance_fields.npz"]
    else:
        field_files = []
    for path in field_files:
        if not path.exists(): continue
        a = np.load(path); fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
        for ax, key, title in zip(axes, ("true_damage", "recovered_damage"), ("Synthetic truth (evaluation only)", "Recovered continuous field")):
            im = ax.scatter(*a["coordinates_mm"].T, c=a[key], s=25, cmap="magma", vmin=0, vmax=.6)
            ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)"); ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, label="Damage D", shrink=.7)
        finish(fig, folder, path.stem)
        residual_path = path.with_name(path.name.replace("_fields", "_residual_0"))
        if residual_path.exists():
            a = np.load(residual_path); fig, ax = plt.subplots(figsize=(7, 4))
            m = a["active_mask"]
            im = ax.scatter(*a["coordinates_mm"][m].T, c=np.linalg.norm(a["residual_mm"][m], axis=1), cmap="viridis")
            ax.set(xlabel="x (mm)", ylabel="y (mm)", title="DIC displacement residual magnitude"); ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, label="mm"); finish(fig, folder, residual_path.stem)
    if case == "multiframe":
        fig, ax = plt.subplots(figsize=(8, 4))
        for path in sorted(folder.glob("forecast_*.npz")):
            a = np.load(path); error = np.sqrt(np.mean((a["predicted_displacement_mm"]-a["true_displacement_mm"])**2, axis=(1, 2)))
            ax.plot(a["cycles"], error, "o-", label=path.stem)
        ax.axvspan(20., 40., alpha=.08, color="C3", label="future after calibration N≤20")
        ax.set(xlabel="Physical cycles N", ylabel="Exact displacement RMSE (mm)", title="Prefix calibration and withheld future evaluation")
        ax.legend(fontsize=8); finish(fig, folder, "multiframe_future_holdout")
