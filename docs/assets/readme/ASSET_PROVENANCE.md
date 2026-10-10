# README scientific figure provenance

These are project-generated, lightweight visualizations of existing synthetic evidence, released with the repository's MIT License. No third-party paper images are used, and no new simulations or independent validation were performed to make these figures. Each PNG has an editable SVG counterpart. `architecture.dot` and `architecture.mmd` have the same nodes/edges as the rendered schematic.

Replot from the repository root:

```bash
MPLCONFIGDIR=/tmp/fatiguedic-readme-mpl python -W error scripts/plot_readme_figures.py
```

The script uses [figure_data.json](figure_data.json): exact numeric extracts plus SHA256 and original paths for every source. Complete original `results/` assets are retained externally to Git; their public distribution is not yet established. [ASSET_MANIFEST.json](ASSET_MANIFEST.json) records the lightweight file hashes (excluding the manifest itself).

| Figure | Authoritative source | What is shown / limitations |
|---|---|---|
| `architecture` | Phase 0–2F source contracts and reports | Conceptual workflow; separate FE nodes/elements/quadrature/DIC; truth isolation; no trained RL policy. Generated DOT/Mermaid graph specifications share the PNG/SVG nodes and edges; layouts may differ by renderer. |
| `tri3_convergence` | `results/phase1a/verified/convergence/error_table.csv`; [Phase 1A](../../PHASE1A_MECHANICS_VERIFICATION_REPORT.md) | Four manufactured-solution mesh levels, exact relative L2 errors vs maximum edge in mm. O(h²)/O(h) reference slopes are guides, not data. Final rates 2.129533/1.055175. Smooth elasticity only; not mesh-objective local damage. |
| `damage_evolution` | `results/phase1b/final/localized_damage/trajectory/truth/trajectory.npz`; [Phase 1B](../../PHASE1B_FATIGUE_EVOLUTION_REPORT.md) | Actual initial/final damage and cumulative increment N0→40 on 16 elements/16 centroid quadrature points. Flat colors preserve the quadrature-to-element mapping; no fabricated nodal damage interpolation. Coordinates mm; damage dimensionless. |
| `damage_identifiability`, `identifiability_spectrum` | `results/phase1c/verified/gaussian_sensitivity/gaussian_sensitivity.npz` and `gaussian_observations/observation.npz`; [Phase 1C](../../PHASE1C_IDENTIFIABILITY_REPORT.md) | Euclidean norms of the two displacement derivatives per normalized parameter at 64 DIC coordinates; original whitened singular values. No rescaling to manufacture observability. Numerical rank six does not imply six accurately estimated parameters. |
| `phase2c_structural_benchmark` | `results/phase2c/20261008T233840Z/final_audit_v4/fair_records.json`; [Phase 2C, section K](../../PHASE2C_PHYSICS_CORRECTION_REPORT.md) | Every one of five paired structural cases 84000–84004, B0/B2/B5, N220 damage RMSE. Bars are means, dots are all case values. Assimilation N≤20; σux=2e−7 mm (0.2 nm), synthetic high SNR. Same law family, not five independent material mechanisms. Means .0196013741/.00201021020/.000350084514. |
| `phase2f_information_rank` | `results/phase2f/20261010T010645Z/development_v2/diagnostics/summary.json`, `static.variants`; [Phase 2F](../../PHASE2F_INFORMATION_SUFFICIENCY_REPORT.md) | Data-only numerical ranks 1/3, 2/3, 2/3, 3/3 for d0/E0/shared load scale. Conditional on externally certified D=0 healthy state, modulus transfer and calibrated shared loading; no fatigue-law identification or formal 100-world accuracy claim. |

The supplied README ZIP had 11 safe entries and valid CRCs. Its original Phase 1A PNG exactly matched the authoritative server PNG. All originals and both README drafts were backed up before integration. Final figures were regenerated from the numeric sources above to provide consistent labels, complete plotting sources, all five benchmark dots and explicit conditional captions. Package figure drafts remain in the excluded publication recovery directory; the authorized original ZIP was deleted after successful integration and hash checks.
