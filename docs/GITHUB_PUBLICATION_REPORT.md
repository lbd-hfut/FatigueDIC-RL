# GitHub Publication Report — FatigueDIC-RL

Publication date: 2026-10-10 (actual consolidated import date). This is a source/documentation publication, not a new scientific qualification. **VALIDATION_READINESS_PARTIALLY_QUALIFIED and RL NO-GO remain unchanged.** No new scientific algorithm, neural training, formal 100-world test or RL run was started.

## A. Actual Git commits and annotated tags

| Logical version | Actual commit | Annotated tag | Files / canonical source SHA256 |
|---|---|---|---|
| Phase 0–2E | `c52c635599d5b3fb7e24b61c54a9377f4c9ceede` | `scientific-baseline-phase2e-partially-qualified` | 314 / `0f98b2593cf9f73c6b03eef462f736aeaab52cf9720f3cd986fca9224e9a2ae5` |
| Phase 2F increment | `154e70eef2d857afc26e7882c341549e3984b1cb` | `scientific-baseline-phase2f-pilot-partially-qualified` | 327 / `9083c806cbd0c141d45c16e88a462ce5f7ee16aefdba1f24d9b74fb60da011c2` |
| Documentation + Validation Readiness | The publication commit introducing this report, message below | No additional qualification tag | 363-file publication payload; a new full SHA is recorded in the external final manifest/receipt |

Exact messages:

```text
chore: freeze verified Phase 0–2E scientific source baseline
research: add development-only Phase 2F information sufficiency pilot
docs: publish FatigueDIC-RL methods and verified research results
```

The first two commits were made at 2026-10-10T11:04:43+08:00, with the configured repository author, not historical phase timestamps. Parent history is the initial `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`, then the two consolidated imports and this documentation increment. Tag object IDs: Phase 2E `84e56d7eb0d7af389df021974a9741f557907de8`, Phase 2F `a6bd960e13932b1108d5d26d6ae9f3ae5b53a2d2`. Both are annotated, with canonical source SHA, original run, limited qualification and RL NO-GO.

A report cannot contain its own final commit/content hash without changing that hash. The exact documentation commit ID and **new full canonical publication SHA256** are therefore stored in `results/github_publication/20261010T030105Z/final_source_verified/audit.json` and `publication_receipt.json`, and reported in the final delivery. The README/CI/ignore changes are never assigned an old 2E/2F SHA. This document's introduction commit can be inspected with:

```bash
git log -1 --diff-filter=A --format='%H %s' -- docs/GITHUB_PUBLICATION_REPORT.md
```

## B. Remote and branch verification

Verified origin: `git@github.com:lbd-hfut/FatigueDIC-RL.git`; GitHub public API identifies [lbd-hfut/FatigueDIC-RL](https://github.com/lbd-hfut/FatigueDIC-RL), public, MIT, default branch `main`. The remote initially contained only initial `main`, no development branch or conflicting tags. Target is **feature/phase0-architecture**. No force push, tag overwrite or merge into main was used.

The installed `gh` credential returned HTTP 401; no credentials were read, replaced or disclosed. Public unauthenticated API verified identity. System SSH config ownership prevented the default client invocation; per-command `ssh -F /dev/null` avoids modifying the system. A port-22 push timed out without creating refs. The publication transport uses [GitHub's official SSH-over-443 method](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port), keeping origin unchanged, checking the existing trusted `github.com` host key via `HostKeyAlias`, and disabling host-key-file updates. Private SSH/system configuration files were not rewritten.

## C. Submitted file manifest and artifact integrity

[PUBLICATION_FILES.txt](reproducibility/PUBLICATION_FILES.txt) is the exact 363-file allowlist. Phase 2E and Phase 2F complete manifests are [phase2e_source_manifest.json](reproducibility/phase2e_source_manifest.json) and [phase2f_source_manifest.json](reproducibility/phase2f_source_manifest.json). Each canonical SHA is SHA256 of the sorted compact JSON mapping `relative_path → file_SHA256`. Every staged historical blob was checked before its commit; the whole frozen tree, not just changed files, had to match.

Both archived baselines and the 336-file readiness snapshot were restored into separate empty temporary directories and every path/content hash verified again. The readiness prepublication source SHA is `7a478794a750afa7dc48f99021b1d003c7c61b7a87ad597a8acc103b2dc06983`. Original recovery statements, reports, failed cases and numerical assets were not overwritten.

The final documentation/readiness delta has **36 new files and 3 modified files** relative to Phase 2F: the nine previously developed readiness files, bilingual README, seven PNG/SVG figure pairs plus editable graph/data/provenance files, plotting source, small reproducibility indices/manifests and this report. Only original README, `.gitignore` and CI changed among the 327 historical files; all historical scientific source, tests, configs and reports remain byte-identical.

The audited historical 314-file list explicitly includes five small textual `src/fatigue_dic_rl.egg-info/*` files. They were retained, and only these explicit ignored files were force-staged, to satisfy the required canonical baseline. No indiscriminate `git add -A` was used. Future metadata cleanup should be separately reviewed, preserving immutable tags. Builds in this task used an isolated source copy to avoid changing these tracked metadata files.

`.gitignore` now excludes results, checkpoints, archives, tar/zip, training weights, caches, virtual environments and local credential/editor/agent files. It does not globally ignore JSON/CSV/NPZ fixtures. No raw scientific asset or original ZIP enters any commit. The preexisting Phase 0 report contains a non-secret historical machine path; it is retained for exact source integrity, not required by public usage commands. Source and asset scans found no private keys, recognized live-token patterns or credential-bearing URLs. All new figure assets are <200 KiB each.

## D. README content verification

[English README](../README.md) and [Chinese README](../README_CN.md) cover motivation, architecture, equations, FEM, fatigue law, DIC likelihood, identifiability, prefix belief, Graph+GRU, bounded correction, model selection/design, measured results, installation, repository layout, limitations, reproducibility, citation and MIT licensing.

Equations were checked against `mechanics/tri3.py`, `materials/plane_stress_mm.py`, `fatigue/models.py`, `inverse/forward_adapter.py`, `inference/sequential_map.py`, `networks/parameter_head.py`, `networks/graph_fusion.py` and `correction/{correction_basis,hybrid_damage_law}.py`. In particular, engineering shear is explicit; the real neural head produces a bounded new normalized estimate and records its delta; energy correction bases are saturation powers; SciPy FEM is not claimed differentiable through PyTorch.

FE nodes, elements, quadrature and DIC points are distinct. Priors are not data information, numerical full rank is not stable recovery, plug-in damage is not a nonlinear posterior mean, prediction/filtered/forecast labels differ, and future DIC cannot enter updates/design. Real DIC integration limitations and optional neural dependency/CI skips are explicit. No invented CLI arguments or paper DOI appear.

## E. Figure provenance and ZIP handling

ZIP SHA256: `585ba9a0ba0ac939fce3e69691c798570d518048c7dbde007d6b33feeba1de42`; 392,362 compressed bytes; **11 files**, 511,139 uncompressed bytes. CRCs passed. Every entry was checked for the required package prefix, absolute/traversal paths, symlinks and oversized contents. No unsafe entry was found. Extraction took place in a fresh independent temporary directory.

All 11 expected paths were integrated without keeping the package-prefix directory: two README drafts; architecture PNG/SVG/DOT; Tri3 PNG; correction benchmark PNG/SVG; information rank PNG/SVG; provenance. Old README/ignore/CI and complete extracted originals were recoverably copied into the excluded publication run. Their contents were checked by SHA256 before integration. Existing assets were compared rather than silently overwritten. The package Tri3 PNG matched the authoritative original exactly.

**The exact authorized root ZIP was deleted only after 11/11 integration/hash checks succeeded.** No other ZIP, archive, source or results was deleted. Complete package originals remain in `results/github_publication/20261010T030105Z/package_extracted/`; recoverable prepublication files are in `prepublication/`.

The final assets add actual Phase 1B damage/increment and Phase 1C sensitivity/spectrum. Result charts were regenerated from original numerical records, preserving all five structural cases and conditional rank captions. See [ASSET_PROVENANCE.md](assets/readme/ASSET_PROVENANCE.md), [ASSET_MANIFEST.json](assets/readme/ASSET_MANIFEST.json), [figure_data.json](assets/readme/figure_data.json) and [plot source](../scripts/plot_readme_figures.py). No unlicensed paper images were used. Two repeated figure renders gave identical hashes for all 18 non-manifest assets. PNGs were opened/visually inspected; SVG XML excludes executable or external-image content. Local Markdown/relative link checks passed, with balanced math/fences. DOT and Mermaid share the rendered diagram's node/edge source; README uses the SVG directly.

## F. Scientific statement audit

| Statement | Source / retained scope |
|---|---|
| 514 local tests, no skips | Publication strict regression; does not qualify every scientific hypothesis |
| Tri3 u O(h²), strain O(h), last rates 2.129533 / 1.055175 | Phase 1A CSV; smooth manufactured elasticity, not mesh-objective local damage |
| Real FE–damage coupling, new-state solve and rollback | Phase 1B source/results; synthetic energy-amplitude law |
| E0–D strict ambiguity | Phase 1C; conditional stiffness, not unique arbitrary damage |
| Classical prefix assimilation, original 89/100 coverage | Phase 2A report/follow-up; PARTIALLY_QUALIFIED retained |
| Neural isolated-CPU ~18.5× parameter-belief speedup, worse accuracy than MAP | Phase 2B; not whole-pipeline/GPU/real-material performance |
| Neural A coverage as low as .76 and future D40 .55 | Phase 2B fixed-material noise checks; no posterior calibration guarantee |
| N220 B0/B2/B5 mean RMSE .0196014/.00201021/.000350085 | Phase 2C five paired cases, one structural family, synthetic 0.2nm noise; not real DIC precision |
| Parameter mismatch favors A/m; matched unnecessary correction can degrade | Phase 2C negative and calibration controls |
| Phase 2D structural 6/8 below 80%; B8 not proved better | 160 cases, 40 matched negative controls; no general robustness qualification |
| Phase 2E Fisher 1/8, discrimination 0/8 | 40 independent worlds, 204 arms; arms are not independent specimens |
| Healthy+reaction improves conditional initial-state rank | Phase 2F; transferable modulus/shared load/external healthy D=0 required; fatigue-law identification not guaranteed |
| Formal 100-world validation not executed | Readiness remains PARTIALLY_QUALIFIED; RL NO-GO |

No real-material calibration, validated RL policy, crack path, RUL, mesh-objective localization, universal structural correction or calibrated high-dimensional damage posterior is claimed. Historic failure reports and qualifications are not edited to produce a stronger release claim.

## G. Pytest, build and CI status

`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -W error`: **514 passed, no skips, 490.64 s**. Scientific source did not change during publication. Environment: Python3.13.12, NumPy2.4.4, SciPy1.17.1, Matplotlib3.10.9, pytest9.0.3, torch2.13.0, setuptools80.10.2. Full logs/checksums are in the excluded publication evidence; [PUBLICATION_VALIDATION.json](reproducibility/PUBLICATION_VALIDATION.json) contains the portable summary.

README commands were actually run with fresh output directories: Tri3 patch PASS; coupled uniform-fatigue PASS; uniform-confounding DIC diagnostic PASSED. Additional constant-hazard verification PASS. `pip wheel --no-deps --no-build-isolation` passed in an independent source copy; wheel was imported from `/tmp`, and numerical imports did not load torch. Wheel SHA256: `4b1951c2f10af92ce5ab83f45d89a371e785aec1d2584872a6968b5d5c6dd222`.

The reviewed CI retains Python3.11/3.13 numerical jobs and adds an explicit CPU-PyTorch complete regression job. It imports torch/the network before pytest, so a missing neural dependency cannot silently qualify that job. Local YAML/workflow structure checks passed. **Final remote CI, Python3.11 and the hosted CPU-PyTorch environment are not claimed independently passed by this document.** Check [Actions](https://github.com/lbd-hfut/FatigueDIC-RL/actions?query=branch%3Afeature%2Fphase0-architecture) and the dated final remote receipt; numerical-only skips are explicit.

## H. Push status and GitHub links

The two exact baselines and their annotated tags were uploaded and checked before committing this publication document. The documentation/readiness payload is the subsequent non-force commit on the same development branch. Its final remote-ref/content verification and exact SHA are recorded in `results/github_publication/20261010T030105Z/post_push_verification.json` and `publication_receipt.json`; these receipts are created only after checking the actual remote, not inferred from a local commit.

- [Development branch](https://github.com/lbd-hfut/FatigueDIC-RL/tree/feature/phase0-architecture)
- [English README](https://github.com/lbd-hfut/FatigueDIC-RL/blob/feature/phase0-architecture/README.md)
- [Chinese README](https://github.com/lbd-hfut/FatigueDIC-RL/blob/feature/phase0-architecture/README_CN.md)
- [Phase 2E annotated baseline](https://github.com/lbd-hfut/FatigueDIC-RL/tree/scientific-baseline-phase2e-partially-qualified)
- [Phase 2F annotated baseline](https://github.com/lbd-hfut/FatigueDIC-RL/tree/scientific-baseline-phase2f-pilot-partially-qualified)
- [Review branch against main](https://github.com/lbd-hfut/FatigueDIC-RL/compare/main...feature/phase0-architecture)

Tag links identify Git tags, not a claim that separate GitHub Release objects were created. The default **main remains initial**; its homepage will show the new README only after an explicitly reviewed PR is merged. No PR or merge is automatically created here.

## I. Remaining limitations

Source publication does not establish independent 100-world validation, experimental material/camera calibration, validated uncertainty, OOD robustness or RL necessity. The full numerical evidence and 161 historical checkpoints remain local external-to-Git research assets; public download/long-term off-machine backup has not been confirmed. Historical `results/` report links intentionally refer to those assets and do not resolve in a source-only GitHub clone. Curated figures, exact small extracts and immutable source manifests are public. Original machine-path provenance is non-secret but is not a portable run requirement. The preserved egg-info snapshot may be modified by future editable builds; build/hash audits should use immutable Git blobs or isolated copies.

## J. Recommended repository maintenance

Review and open a PR from the development branch into main; wait for the explicit neural and numerical CI jobs and inspect scientific limitations before merging. Preserve both annotated tags and PARTIALLY_QUALIFIED/NO-GO records. Maintain small source/config/test/document indices in Git; distribute large results through an independently verified, immutable external asset store and selectively use LFS only for deliberately released checkpoints. Add a locked environment when validated, keep scientific long-run budgets separate from ordinary CI, and record exact source/config/asset hashes per run. Review generated metadata cleanup independently rather than rewriting the frozen tags. Do not start new research, formal100 or RL as a side effect of this publication.
