"""Render curated README evidence; no simulations, fitting or truth-dependent design.

The small JSON contains exact extracts from immutable, hashed scientific assets.
Run from the repository root after installing the optional verification extra.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.tri import Triangulation


def render(output):
    data = json.loads((output / "figure_data.json").read_text())
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "FatigueDIC-RL-readme-v1", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(output / (name + ".png"), dpi=150, bbox_inches="tight")
        svg_path = output / (name + ".svg")
        fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
        # Matplotlib emits harmless trailing spaces inside SVG paths. Keep the
        # editable vector geometry unchanged while passing the Git text audit.
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        plt.close(fig)

    # One declared node/edge specification generates the PNG/SVG, DOT and Mermaid.
    nodes = {
        "geometry": (0, 2, "Geometry / mesh", "FE nodes V; elements E", "#eaf1fb"),
        "fem": (1, 2, "FEM + constitutive law", "Equilibrium; strain / stress at G", "#e5f4ed"),
        "fatigue": (2, 2, "Fatigue evolution", "Hazard; irreversible history at G", "#e5f4ed"),
        "observation": (2, 1, "FE-to-DIC observation", "Shape functions; irregular Q; covariance", "#eaf1fb"),
        "inverse": (1, 1, "Identifiability / inversion", "Whitened Jacobian; SVD; TRF", "#fff2df"),
        "belief": (0, 1, "Sequential belief", "Prefix MAP / optional Graph + GRU", "#eee7f8"),
        "correction": (0, 0, "Physics-guided correction", "A/m refit or bounded low-dimensional law", "#fbe9e5"),
        "forecast": (1, 0, "Future open-loop prediction", "Rebuild history; solve FEM at new D", "#e5f4ed"),
        "scope": (2, 0, "Synthetic research scope", "Truth: generation / evaluation only\nRL NO-GO; no validated crack path / RUL", "#fff2df"),
    }
    edges = [("geometry", "fem"), ("fem", "fatigue"), ("fatigue", "observation"),
             ("observation", "inverse"), ("inverse", "belief"), ("belief", "correction"),
             ("correction", "forecast")]
    fig, ax = plt.subplots(figsize=(12, 7.1))
    width, height = 2.98, .99
    centers = {n: (x * 3.6, y * 1.8) for n, (x, y, *_) in nodes.items()}
    for name, (_, _, title, sub, color) in nodes.items():
        x, y = centers[name]
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                             boxstyle="round,pad=.03,rounding_size=.1", fc=color, ec="#7c8da2", lw=1.2)
        ax.add_patch(box)
        ax.text(x, y+.16, title, ha="center", va="center", weight="bold", fontsize=11)
        ax.text(x, y-.19, sub, ha="center", va="center", fontsize=8.4, linespacing=1.5)
    for a, b in edges:
        ca, cb = np.array(centers[a]), np.array(centers[b]); direction = cb-ca
        fraction = (width/2+.05)/abs(direction[0]) if direction[0] else (height/2+.07)/abs(direction[1])
        start, end = ca + fraction*direction, cb - fraction*direction
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "#4e647c", "lw": 1.7})
    ax.set(xlim=(-1.8, 9), ylim=(-.9, 4.35)); ax.axis("off")
    fig.suptitle("FatigueDIC-RL: DIC-observed physics-guided inference", fontsize=17, y=.99)
    fig.text(.5, .045, "Each parameter candidate rebuilds its own material history. Observed prefixes and future forecasts remain separate.", ha="center", fontsize=10)
    save(fig, "architecture")
    dot = ['digraph G {', '  graph [rankdir=LR];', '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans"];']
    mermaid = ["flowchart LR"]
    for n, (_, _, title, sub, color) in nodes.items():
        label = title + "\n" + sub
        dot.append(f'  {n} [label={json.dumps(label)}, fillcolor="{color}"];')
        mermaid.append(f'  {n}["{title}<br/>{sub.replace(chr(10), "<br/>")}"]')
    for a, b in edges:
        dot.append(f"  {a} -> {b};"); mermaid.append(f"  {a} --> {b}")
    dot.append("}")
    (output / "architecture.dot").write_text("\n".join(dot)+"\n")
    (output / "architecture.mmd").write_text("\n".join(mermaid)+"\n")

    rows = data["convergence"]; h = np.array([r["h_max_mm"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, order, title in zip(axes, ("relative_displacement_L2", "relative_strain_L2"), (2, 1),
                                     ("Displacement: O(h²)", "Strain: O(h)")):
        y = np.array([r[key] for r in rows])
        ax.loglog(h, y, "o-", color="#286b9b", label="Verified manufactured solution")
        ax.loglog(h, y[-1]*(h/h[-1])**order, "--", color="#a25736", label=f"Reference h^{order}")
        ax.set(xlabel="Maximum element edge h (mm)", ylabel="Relative L2 error", title=title)
        ax.grid(True, which="both", alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Tri3 plane-stress verification: smooth manufactured solution", y=1.04)
    fig.tight_layout(); save(fig, "tri3_convergence")

    d = data["damage"]; nodes = np.array(d["nodes_mm"]); tri = Triangulation(*nodes.T, triangles=d["connectivity"])
    assert d["quadrature_to_element"] == list(range(len(d["connectivity"])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True)
    for ax, key, label in zip(axes, ("initial_damage", "final_damage", "accumulated_increment"),
                              ("Initial D at N=0", f"Damage D at N={d['N_final']:g}", "Accumulated ΔD")):
        values = np.array(d[key]); vmax = max(d["final_damage"]) if key != "accumulated_increment" else max(values)
        image = ax.tripcolor(tri, facecolors=values, shading="flat", cmap="magma", vmin=0, vmax=vmax, edgecolors="#777777", linewidth=.35)
        ax.set(title=label, xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
        fig.colorbar(image, ax=ax, shrink=.65, label="Damage (dimensionless)")
    fig.suptitle("Actual coupled FEM–fatigue trajectory: centroid quadrature fields", y=.98)
    fig.text(.5, .02, "16 elements / 16 quadrature points; local damage representation remains mesh sensitive.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.93)); save(fig, "damage_evolution")

    s = data["sensitivity"]; xy = np.array(s["coordinates_mm"]); values = np.array(s["magnitude_mm_per_dimensionless"])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.5), sharex=True, sharey=True)
    for k, ax in enumerate(axes.ravel()):
        image = ax.scatter(*xy.T, c=values[:, k], cmap="viridis", s=20)
        ax.set(title=s["parameter_names"][k], xlabel="x (mm)", ylabel="y (mm)", xlim=(0,2), ylim=(0,1), aspect="equal")
        fig.colorbar(image, ax=ax, label="mm / normalized parameter", shrink=.75)
    fig.suptitle("Gaussian initial damage: DIC displacement sensitivity magnitude", y=1.01)
    fig.tight_layout(); save(fig, "damage_identifiability")
    fig, ax = plt.subplots(figsize=(7, 3.8)); sv = np.array(s["singular_values"])
    ax.semilogy(np.arange(1, len(sv)+1), sv, "o-", color="#286b9b")
    ax.axhline(1, ls="--", color="#a25736", label="Unit-noise scale (diagnostic)")
    ax.set(xlabel="Singular direction", ylabel="Whitened Jacobian singular value", title="Numerical rank 6; only 4 singular values exceed one")
    ax.set_xticks(range(1,7)); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(); save(fig, "identifiability_spectrum")

    fig, ax = plt.subplots(figsize=(8.3, 4.3)); colors = ["#758293", "#2d78a6", "#249477"]
    labels = ["Physics-only B0", "A/m refit B2", "DIC-only K2 B5"]
    for i, (row, color, label) in enumerate(zip(data["structural"], colors, labels)):
        y = np.array(row["damage_rmse_N220"])
        ax.bar(i, y.mean(), width=.6, color=color, alpha=.7)
        ax.scatter(i+np.linspace(-.16,.16,5), y, color="#243241", s=22, zorder=3)
        ax.text(i, y.mean()*1.13, f"{y.mean():.6g}", ha="center", fontsize=9)
    ax.set(yscale="log", ylabel="Damage RMSE at N=220 (dimensionless)", title="Structural mismatch: five paired test cases", ylim=(1e-4,5e-2))
    ax.set_xticks(range(3), labels); ax.grid(axis="y", alpha=.2)
    fig.text(.5,.015,"Synthetic σux=2×10⁻⁷ mm (0.2 nm); prefix N≤20; bars: means; dots: all five cases.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1)); save(fig, "phase2c_structural_benchmark")

    fig, ax = plt.subplots(figsize=(9, 4)); ranks = [r["rank"] for r in data["initial_state_ranks"]]
    ax.bar(range(4), ranks, color=["#758293", "#2d78a6", "#7b68a6", "#249477"], width=.6)
    ax.set_xticks(range(4), ["DIC", "DIC + reaction", "DIC + healthy", "DIC + both"])
    ax.set(ylim=(0,3.6), ylabel="Data-only numerical rank / 3", title="Initial state: unknown d0, E0 and shared loading scale")
    ax.set_yticks(range(4)); ax.grid(axis="y", alpha=.2)
    for i, value in enumerate(ranks): ax.text(i, value+.1, f"{value}/3", ha="center", weight="bold")
    fig.text(.5,.015,"Conditional on transferable E0, externally certified healthy D=0 and shared calibrated loading.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1)); save(fig, "phase2f_information_rank")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("docs/assets/readme"))
    render(parser.parse_args().assets)
