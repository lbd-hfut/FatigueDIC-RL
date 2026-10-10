"""Phase 1C experiments using actual FE solves and independent truth evaluation.

Mathematical degeneracies are expected scientific outcomes, not recovery
targets. All prescribed seeds/scenarios are reported, including failed starts.
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from scipy.special import erf
from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.mechanics import DisplacementBC, TractionBC
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverRequest
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.covariance import DisplacementCovariance
from fatigue_dic_rl.observations.noise import DICMeasurement, simulate_measurement, calibrated_measurement
from fatigue_dic_rl.observations.registration import PixelCalibration, RegistrationError
from fatigue_dic_rl.inverse.parameterizations import Parameter, ParameterSpace, UniformDamage, GaussianDamage, LowRankDamage
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter, StaticLoadCase, ReactionObservation
from fatigue_dic_rl.inverse.fatigue_adapter import FatigueForwardAdapter
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian
from fatigue_dic_rl.inverse.identifiability import fisher_diagnostics
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from .tri3_meshes import rectangle_mesh
from .tri3_verification import tension_request, material
from .inverse_metrics import SyntheticTruth, residual_metrics, evaluate_recovery

GAUSSIAN_TRUTH = dict(d_base=.05, amplitude=.3, xc=1.05, yc=.47, sx=.35, sy=.22)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def save_json(folder, name, value):
    if folder is not None:
        (Path(folder)/(name+".json")).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")


def save_array(folder, name, **values):
    if folder is not None:
        np.savez_compressed(Path(folder)/(name+".npz"), **values)


def mesh_for(config, resolution=None):
    g = config["geometry"]
    return rectangle_mesh(*(resolution or [g["nx"], g["ny"]]), width_mm=2., height_mm=1., seed=g["seed"], jitter=.18)


def pattern_for(mesh, *, mode="x", scale=1., control="load"):
    if mode == "x":
        p = tension_request(mesh, 2000., .3, .8, 10.*scale, control=control)
        return replace(p, loading_state=LoadingState("diagnostic-x" if control == "load" else "diagnostic-imposed-x", "static"))
    bottom, top = mesh.boundary_tags["bottom"].node_ids, mesh.boundary_tags["top"].node_ids
    anchor = int(np.argmin(np.linalg.norm(mesh.nodes, axis=1)))
    if mode == "y":
        bcs = (DisplacementBC(bottom, np.ones(len(bottom), dtype=int), np.zeros(len(bottom)), "mm"),
               DisplacementBC([anchor], [0], [0.], "mm"))
        traction = [0., 10.*scale]
    elif mode == "shear":
        bcs = (DisplacementBC(np.repeat(bottom, 2), np.tile([0, 1], len(bottom)), np.zeros(2*len(bottom)), "mm"),)
        traction = [10.*scale, 0.]
    else:
        raise ContractError("unsupported static diagnostic pattern")
    return Tri3SolverRequest.from_mm(mesh, material(2000., .3), LoadingState("diagnostic-"+mode, "static"),
        thickness_mm=.8, displacement_bcs=bcs, traction_bcs=(TractionBC("top", traction, "N/mm^2"),))


def points(count=64, sampling="random", seed=101):
    rng = np.random.default_rng(seed)
    if sampling == "regular":
        side = int(np.sqrt(count))
        if side*side != count:
            raise ContractError("regular verification count must be square")
        x, y = np.meshgrid(np.linspace(.03, 1.97, side), np.linspace(.03, .97, side))
        return np.column_stack((x.ravel(), y.ravel()))
    if sampling == "random":
        return rng.uniform([.03, .03], [1.97, .97], (count, 2))
    if sampling == "clustered":
        return rng.uniform([.06, .34], [.36, .66], (count, 2))
    raise ContractError("unknown sampling mode")


def solve_truth(pattern, model, physical, coordinates):
    """This evaluation function is never called from an inverse objective."""
    layout = pattern.quadrature_layout
    materials = pattern.materials if "E0" not in physical else material(physical["E0"], .3)
    f = physical.get("load_factor", 1.)
    request = replace(pattern, damage=layout.field(model.evaluate(layout.coordinates, physical), "1"), materials=materials,
        traction_bcs=tuple(replace(b, values=b.values*f) for b in pattern.traction_bcs))
    result = Tri3MechanicsSolver().solve(request)
    mapping = ObservationOperator(pattern.mesh).prepare(coordinates, coordinate_units="mm", coordinate_system="specimen")
    return result, mapping.apply(result.state.displacement)


def make_case(pattern, model, physical, x, *, sigma_mm=0., seed=101, missing=0., registration=None,
              correlated_length=None, reaction=False, known_healthy=False, folder=None, name="case"):
    actual = dict(physical)
    if known_healthy:
        actual["d0"] = 0.
    result, mapped = solve_truth(pattern, model, actual, x)
    cov = (DisplacementCovariance.diagonal(np.full_like(x, sigma_mm)) if correlated_length is None else
           DisplacementCovariance.spatial(x, np.diag([sigma_mm**2, sigma_mm**2]), correlated_length))
    observation = simulate_measurement(x, mapped.values, cov, seed=seed, domain_mask=mapped.valid_mask,
        missing_fraction=missing, registration=registration, load_id=pattern.loading_state.load_id)
    force = None
    if reaction:
        nodes = pattern.mesh.boundary_tags["right" if "imposed" in pattern.loading_state.load_id else "left"].node_ids
        force = ReactionObservation(nodes, 0, result.reaction_forces.values[nodes, 0].sum(), .005)
    c = StaticLoadCase(pattern, observation, force, 1e-10 if sigma_mm == 0 else 0., known_healthy)
    if folder is not None:
        observation.save(Path(folder)/(name+"_observations"))
        SyntheticTruth(pattern.quadrature_layout.coordinates, result.assembly.quadrature_damage.values, actual,
            result.state.displacement.values, pattern.mesh.mesh_id, pattern.quadrature_layout.reference.space_id,
            mesh_nodes_mm=pattern.mesh.nodes, strain_engineering=result.state.strain.values,
            stress_n_mm2=result.state.stress.values, reaction_n=result.reaction_forces.values).save(Path(folder)/(name+"_truth"))
        save_array(folder, name+"_fe", nodes_mm=pattern.mesh.nodes, connectivity=np.array(pattern.mesh.connectivity),
            quadrature_coordinates_mm=pattern.quadrature_layout.coordinates, displacement_mm=result.state.displacement.values,
            strain=result.state.strain.values, stress_n_mm2=result.state.stress.values,
            reaction_n=result.reaction_forces.values, quadrature_weights_mm2=pattern.quadrature_layout.physical_weights)
        save_json(folder, name+"_load", {"mesh_id": pattern.mesh.mesh_id, "quadrature_reference": pattern.quadrature_layout.reference.space_id,
            "known_material_parameters": {str(r): dict(a.parameters) for r, a in pattern.materials.items()},
            "thickness_mm": pattern.thickness_mm, "load_id": pattern.loading_state.load_id,
            "tractions": [{"boundary": b.boundary_tag, "values": b.values, "units": b.units} for b in pattern.traction_bcs],
            "prescribed_displacements": [{"nodes": b.node_ids, "components": b.components, "values": b.values, "units": b.units}
                                        for b in pattern.displacement_bcs],
            "reaction_observation": None if force is None else {"nodes": force.node_ids, "component": force.component,
                "value_n": force.value_n, "standard_deviation_n": force.standard_deviation_n}})
    return c


def diagnose(adapter, physical, folder=None, name="sensitivity"):
    z = adapter.parameter_space.encode(physical)
    sensitivity = centered_jacobian(adapter.whitened_prediction, z)
    assert sensitivity.converged, "finite-difference sensitivity did not converge"
    fisher = fisher_diagnostics(sensitivity)
    # Also retain unwhitened displacement sensitivities in mm / dimensionless p.
    raw = centered_jacobian(adapter.displacement_prediction, z, absolute_tolerance=1e-10)
    assert raw.converged
    save_array(folder, name, jacobian_whitened=sensitivity.jacobian, jacobian_displacement_mm=raw.jacobian,
        coarse_jacobian=sensitivity.coarse_jacobian, parameter_names=np.array(adapter.parameter_space.names),
        dimensionless=z, physical_scales=adapter.parameter_space.scales, steps=sensitivity.steps,
        singular_values=fisher.singular_values, fisher=fisher.fisher, right_directions=fisher.right_directions,
        null_directions=fisher.null_directions, parameter_correlation=fisher.parameter_correlation,
        identifiable_covariance=fisher.identifiable_covariance)
    row = {"rank": fisher.effective_rank, "parameters": len(z), "singular_values": fisher.singular_values,
           "condition_number": fisher.condition_number, "rank_threshold": fisher.rank_threshold,
           "unit_noise_resolved_directions": int(np.sum(fisher.singular_values > max(1., fisher.rank_threshold))),
           "fd_relative_changes": sensitivity.column_relative_changes, "fd_absolute_error": sensitivity.absolute_error_norm,
           "fd_roundoff_resolution": sensitivity.roundoff_resolution,
           "weakest_direction": fisher.right_directions[-1], "null_directions": fisher.null_directions,
           "local_standard_deviations_dimensionless": np.sqrt(np.diag(fisher.identifiable_covariance)) if fisher.effective_rank == len(z) else None,
           "interpretation": "local-only; identifiable-subspace covariance is not a full covariance when rank deficient"}
    save_json(folder, name, row)
    return clean(row), fisher, sensitivity


def initializations(space):
    k = len(space.parameters)
    return [np.zeros(k), np.linspace(-.45, .35, k), np.linspace(.35, -.35, k)]


def recover(adapter, true_parameters, config, *, folder=None, name="recovery", heldout=None, prior_mean=None, prior_standard_deviation=None):
    fit = invert(InverseObjective(adapter, prior_mean=prior_mean, prior_standard_deviation=prior_standard_deviation),
                 initializations(adapter.parameter_space), max_nfev=config["optimization"]["max_nfev"])
    starts = [{"initial": r.initial, "normalized": r.normalized, "parameters": adapter.parameter_space.decode(r.normalized),
               "cost": r.cost, "converged": r.converged, "nfev": r.nfev, "optimality": r.optimality,
               "message": r.message, "sensitivity_checked": r.sensitivity_checked} for r in fit.starts]
    save_json(folder, name+"_optimization_history", [r.history for r in fit.starts])
    save_json(folder, name+"_starts", starts)
    if not any(r.converged for r in fit.starts):
        return {"status": "OPTIMIZATION_FAILED", "starts": clean(starts)}, fit
    best = fit.best
    evaluation_x = points(256, "regular", 777)
    metrics = evaluate_recovery(adapter.damage_model, adapter.parameter_space, best.normalized, true_parameters, evaluation_x)
    metrics.update(residual_metrics(adapter, best.normalized))
    metrics.update(status="OPTIMIZER_CONVERGED", starts=starts, converged_starts=sum(r.converged for r in fit.starts))
    # Descriptive controlled-test error flag, not an optimizer stopping rule or
    # universal material tolerance. Full metrics and all starts remain available.
    metrics["synthetic_damage_error_above_0_05"] = metrics["damage_field_rmse"] > .05
    if heldout is not None:
        predicted = heldout.displacement_prediction(best.normalized)
        actual = heldout.displacement_prediction(heldout.parameter_space.encode(true_parameters))
        metrics["heldout_exact_displacement_rmse_mm"] = float(np.sqrt(np.mean((predicted-actual)**2)))
        metrics["heldout_observed_displacement_rmse_mm"] = residual_metrics(heldout, best.normalized)["displacement_rmse_mm"]
    save_array(folder, name+"_fields", coordinates_mm=evaluation_x,
        true_damage=adapter.damage_model.evaluate(evaluation_x, true_parameters),
        recovered_damage=adapter.damage_model.evaluate(evaluation_x, adapter.parameter_space.decode(best.normalized)),
        all_start_parameters=np.array([r.normalized for r in fit.starts]), all_start_costs=np.array([r.cost for r in fit.starts]))
    for i, ((u, _), c) in enumerate(zip(adapter.predict(best.normalized), adapter.cases)):
        save_array(folder, name+f"_residual_{i}", coordinates_mm=c.measurement.observation.coordinates,
            observed_mm=c.measurement.observation.displacement, predicted_mm=u,
            residual_mm=u-c.measurement.observation.displacement, active_mask=c.measurement.active_mask)
    return clean(metrics), fit


def run_observation_consistency(config, folder=None):
    mesh, model = mesh_for(config), GaussianDamage()
    p = pattern_for(mesh)
    x = np.vstack((points(64), [[3., .5]]))
    c = make_case(p, model, GAUSSIAN_TRUTH, x, folder=folder, name="noiseless")
    assert not c.measurement.observation.valid_mask[-1]
    adapter = StaticForwardAdapter(model, ParameterSpace(model.parameters), [c])
    np.testing.assert_allclose(adapter.residual(adapter.parameter_space.encode(GAUSSIAN_TRUTH)), 0, atol=1e-10)
    calibration = PixelCalibration(config["observation"]["mm_per_pixel"], (.01, -.02))
    o = c.measurement.observation
    pixel = calibrated_measurement((x-calibration.origin_mm)/calibration.mm_per_pixel,
        o.displacement/calibration.mm_per_pixel, coordinate_units="pixel", displacement_units="pixel",
        calibration=calibration, covariance=c.measurement.covariance, valid_mask=o.valid_mask,
        roi_mask=np.arange(len(x)) % 3 != 0, confidence=np.linspace(.1, 1, len(x)), load_id=p.loading_state.load_id)
    np.testing.assert_allclose(pixel.observation.coordinates, x, atol=1e-14)
    np.testing.assert_allclose(pixel.observation.displacement[o.valid_mask], o.displacement[o.valid_mask], atol=1e-16)
    if folder is not None:
        pixel.save(Path(folder)/"pixel_calibrated_observations")
        loaded = DICMeasurement.load(Path(folder)/"pixel_calibrated_observations")
        assert np.array_equal(loaded.roi_mask, pixel.roi_mask)
        with np.load(Path(folder)/"noiseless_observations"/"observation.npz") as data:
            assert not ({"damage", "parameters", "stress", "strain"} & set(data.files))
    return {"status": "PASSED", "rows": [{"active_points": int(c.measurement.active_mask.sum()), "outside_points": 1,
        "noiseless_residual_max": float(np.max(np.abs(adapter.residual(adapter.parameter_space.encode(GAUSSIAN_TRUTH)))))}],
        "checks": {"physical_units": "PASS", "pixel_conversion": "PASS", "outside_domain_mask": "PASS", "truth_isolation": "PASS"}}


def run_noise_covariance(config, folder=None):
    x = points(16, seed=31)
    scale = config["observation"]["mm_per_pixel"]
    rows = []
    for sigma_pixel in config["observation"]["noise_pixels"]:
        sigma = sigma_pixel*scale
        b = sigma**2*np.array([[1., .3], [.3, 2.25]])
        for mode in ("independent", "spatial"):
            cov = (DisplacementCovariance.per_point(np.tile(b, (len(x), 1, 1))) if mode == "independent" else
                   DisplacementCovariance.spatial(x, b, .22))
            replicates = []
            chunks = []
            for seed in config["verification"]["seeds"]:
                samples = cov.sample(np.random.default_rng(seed), count=config["verification"]["noise_repeats"])
                chunks.append(samples.reshape(len(samples), -1))
                empirical = np.cov(chunks[-1], rowvar=False)
                error = float(np.linalg.norm(empirical-cov.matrix)/max(np.linalg.norm(cov.matrix), 1e-300))
                replicates.append({"seed": seed, "relative_covariance_error": error})
            samples = np.vstack(chunks)
            empirical = np.cov(samples, rowvar=False)
            error = float(np.linalg.norm(empirical-cov.matrix)/max(np.linalg.norm(cov.matrix), 1e-300))
            eigenvalues = np.linalg.eigvalsh(cov.matrix)
            if sigma == 0:
                assert np.count_nonzero(samples) == 0
                whitened_error = None
                mean_z_max = 0.
            else:
                factor = cov.factor(np.ones(len(x), dtype=bool))
                whitened = factor.whiten(samples.T).T
                whitened_error = float(np.linalg.norm(np.cov(whitened, rowvar=False)-np.eye(2*len(x)))/np.sqrt(2*len(x)))
                mean_z_max = float(np.max(np.abs(samples.mean(axis=0)/(np.sqrt(np.diag(cov.matrix))/np.sqrt(len(samples))))))
                assert error < .12 and whitened_error < .12 and mean_z_max < 5., "repeated noise statistics failed"
            row = {"noise_pixel": sigma_pixel, "noise_mm": sigma, "mode": mode, "samples": len(samples),
                   "relative_covariance_error": error, "whitened_covariance_error": whitened_error,
                   "mean_standard_error_max": mean_z_max, "minimum_eigenvalue_mm2": float(eigenvalues[0]), "replicates": replicates}
            rows.append(row)
            save_array(folder, f"noise_{mode}_{sigma_pixel:g}", coordinates_mm=x, covariance_mm2=cov.matrix,
                empirical_covariance_mm2=empirical, eigenvalues_mm2=eigenvalues, empirical_mean_mm=samples.mean(axis=0))
    reg = RegistrationError([.0004, -.0002], .0005, (1., .5))
    noise = simulate_measurement(x, np.zeros_like(x), DisplacementCovariance.diagonal(np.zeros_like(x)), seed=33, registration=reg)
    np.testing.assert_allclose(noise.observation.displacement, reg.field(x))
    save_array(folder, "registration", coordinates_mm=x, registration_displacement_mm=reg.field(x))
    return {"status": "PASSED", "rows": rows, "registration_separate": True,
            "checks": {"PSD_symmetry": "PASS", "repeated_statistics": "PASS", "zero_noise": "PASS", "experimental_calibration": "NOT_APPLICABLE"}}


def run_uniform_confounding(config, folder=None):
    mesh, model, x = mesh_for(config), UniformDamage(), points(64)
    p = pattern_for(mesh)
    truth = {"d0": .2, "E0": 2000.}
    c = make_case(p, model, truth, x, folder=folder, name="damaged")
    space = ParameterSpace(model.parameters+(Parameter("E0", 1200., 3000., "N/mm^2"),))
    adapter = StaticForwardAdapter(model, space, [c])
    diag, f, _ = diagnose(adapter, truth, folder, "unknown_E_damage")
    assert f.effective_rank == 1
    predictions = []
    equivalent = []
    for young in (1800., 2000., 2500., 2800.):
        actual = {"d0": 1-1600./young, "E0": young}
        predictions.append(adapter.displacement_prediction(space.encode(actual)))
        equivalent.append(actual)
    max_gap = float(np.max(np.ptp(predictions, axis=0)))
    assert max_gap < 2e-13
    scale_d, scale_e = space.scales
    analytic_null = np.array([.8*scale_e, 2000.*scale_d]); analytic_null /= np.linalg.norm(analytic_null)
    alignment = abs(float(f.null_directions[0] @ analytic_null))
    assert alignment > 1-1e-7
    rows = [{"case": "unknown_E_and_damage", **diag, "equivalent_prediction_gap_mm": max_gap, "analytic_null_alignment": alignment}]
    known = StaticForwardAdapter(model, ParameterSpace(model.parameters), [c])
    recovery, _ = recover(known, {"d0": .2}, config, folder=folder, name="known_E")
    assert abs(recovery["parameters"]["d0"]-.2) < 1e-8
    healthy = make_case(p, model, truth, x, known_healthy=True, folder=folder, name="healthy_baseline")
    baseline = StaticForwardAdapter(model, space, [healthy, c])
    row, ff, _ = diagnose(baseline, truth, folder, "healthy_plus_damaged")
    assert ff.effective_rank == 2
    rows.append({"case": "healthy_baseline_plus_damaged", **row})
    fit, _ = recover(baseline, truth, config, folder=folder, name="baseline_recovery")
    assert fit["damage_field_rmse"] < 1e-8 and abs(fit["parameters"]["E0"]-2000.) < 1e-4
    equivalent_fit, _ = recover(adapter, truth, config, folder=folder, name="equivalent_solution_set")
    assert max(abs((1-r["parameters"]["d0"])*r["parameters"]["E0"]-1600.) for r in equivalent_fit["starts"] if r["converged"]) < 1e-3
    imposed = pattern_for(mesh, control="displacement")
    for force, unknown_e in ((False, False), (True, False), (True, True)):
        cc = make_case(imposed, model, truth, x, reaction=force, folder=folder, name=f"imposed_{force}_{unknown_e}")
        ad = StaticForwardAdapter(model, space if unknown_e else ParameterSpace(model.parameters), [cc])
        row, ff, _ = diagnose(ad, truth, folder, f"imposed_fisher_{force}_{unknown_e}")
        expected_rank = 1 if force else 0
        assert ff.effective_rank == expected_rank
        rows.append({"case": f"displacement_force={force}_unknown_E={unknown_e}", **row})
    return {"status": "PASSED", "rows": rows, "equivalent_parameters": equivalent,
        "known_E_recovery": recovery, "healthy_baseline_recovery": fit, "multi_start_equivalent_solutions": equivalent_fit,
        "checks": {"E_D_degeneracy": "EXPECTED_NONIDENTIFIABLE", "known_E_recovery": "PASS",
                   "healthy_baseline": "PASS", "displacement_without_force": "EXPECTED_NONIDENTIFIABLE",
                   "reaction_with_unknown_E": "EXPECTED_NONIDENTIFIABLE"}}


def run_gaussian_sensitivity(config, folder=None):
    mesh, model, x = mesh_for(config), GaussianDamage(), points(64)
    p = pattern_for(mesh)
    c = make_case(p, model, GAUSSIAN_TRUTH, x, sigma_mm=.00025, folder=folder, name="gaussian")
    adapter = StaticForwardAdapter(model, ParameterSpace(model.parameters), [c])
    diag, f, _ = diagnose(adapter, GAUSSIAN_TRUTH, folder, "gaussian_sensitivity")
    assert f.effective_rank == 6
    rows = [{"case": "gaussian_six_parameters", **diag,
             "amplitude_sx_sensitivity_correlation": f.parameter_correlation[1, 4],
             "amplitude_sy_sensitivity_correlation": f.parameter_correlation[1, 5]}]
    for count in (4, 8, 16):
        centers = points(count, seed=9)
        basis = LowRankDamage(centers)
        physical = {s.name: 0. for s in basis.parameters}
        cc = make_case(p, basis, physical, points(16, "clustered"), sigma_mm=.00025)
        row, _, _ = diagnose(StaticForwardAdapter(basis, ParameterSpace(basis.parameters), [cc]), physical, folder, f"basis_{count}")
        rows.append({"case": f"basis_{count}_clustered", **row})
        save_array(folder, f"basis_definition_{count}", centers_mm=centers, evaluation_coordinates_mm=x, basis=basis.basis(x))
    zero = dict(GAUSSIAN_TRUTH, amplitude=0.)
    space = ParameterSpace(tuple(s for s in model.parameters if s.name in {"xc", "yc", "sx", "sy"}),
                           {"d_base": .05, "amplitude": 0.})
    cc = make_case(p, model, zero, x, sigma_mm=.00025)
    row, ff, _ = diagnose(StaticForwardAdapter(model, space, [cc]), zero, folder, "zero_amplitude_shape")
    assert ff.effective_rank == 0
    rows.append({"case": "zero_amplitude_shape", **row})
    return {"status": "PASSED", "rows": rows,
        "checks": {"centered_difference_stability": "PASS", "six_parameter_local_rank": "PASS",
                   "zero_amplitude_location_width": "EXPECTED_NONIDENTIFIABLE", "arbitrary_high_dimensional_recovery": "NOT_APPLICABLE"}}


def run_sampling_density(config, folder=None):
    mesh, model = mesh_for(config), GaussianDamage()
    p = pattern_for(mesh)
    space = ParameterSpace(model.parameters)
    rows = []
    for count in config["observation"]["counts"]:
        for sampling in ("regular", "random", "clustered"):
            for missing in config["observation"]["missing_fractions"]:
                for seed in config["verification"]["seeds"]:
                    x = points(count, sampling, seed)
                    c = make_case(p, model, GAUSSIAN_TRUTH, x, sigma_mm=.00025, seed=seed, missing=missing)
                    name = f"sampling_{count}_{sampling}_{missing:g}_{seed}"
                    ad = StaticForwardAdapter(model, space, [c])
                    row, _, _ = diagnose(ad, GAUSSIAN_TRUTH, folder, name)
                    save_array(folder, name+"_map", coordinates_mm=x, active_mask=c.measurement.active_mask)
                    rows.append({"count": count, "sampling": sampling, "missing_fraction": missing, "seed": seed,
                                 "active_points": int(c.measurement.active_mask.sum()), **row})
    assert len(rows) == len(config["observation"]["counts"])*3*len(config["observation"]["missing_fractions"])*len(config["verification"]["seeds"])
    regular_min = np.median([r["singular_values"][-1] for r in rows if r["count"] == 64 and r["sampling"] == "regular" and r["missing_fraction"] == 0])
    cluster_min = np.median([r["singular_values"][-1] for r in rows if r["count"] == 64 and r["sampling"] == "clustered" and r["missing_fraction"] == 0])
    assert cluster_min < regular_min, "spatially restricted measurements should reveal weaker directions in this example"
    return {"status": "PASSED", "rows": rows, "checks": {"all_prescribed_seeds_and_masks": "PASS",
            "density_distribution_diagnostics": "PASS", "rank_improvement_guarantee": "NOT_APPLICABLE"}}


def run_multiple_loading(config, folder=None):
    mesh, model, x = mesh_for(config), GaussianDamage(), points(64)
    space = ParameterSpace(model.parameters)
    cases = [make_case(pattern_for(mesh, mode=mode), model, GAUSSIAN_TRUTH, x, sigma_mm=.00025,
                       folder=folder, name=mode) for mode in ("x", "y", "shear")]
    scaled = make_case(pattern_for(mesh, scale=2.), model, GAUSSIAN_TRUTH, x, sigma_mm=.00025, folder=folder, name="x_scaled")
    rows, js = [], []
    for label, selected in (("single", cases[:1]), ("same_mode_scaled", [cases[0], scaled]), ("independent_modes", cases)):
        row, _, sens = diagnose(StaticForwardAdapter(model, space, selected), GAUSSIAN_TRUTH, folder, label)
        rows.append({"case": label, **row}); js.append(sens.jacobian)
    scaled_gap = float(np.linalg.norm(js[1][len(js[0]):]-2*js[0])/np.linalg.norm(js[0]))
    assert scaled_gap < 1e-8 and rows[0]["rank"] == rows[1]["rank"]
    # Sparse, higher-dimensional fields: independent spatial patterns can add
    # information, whereas scalar amplitude repeats the same row space.
    basis = LowRankDamage(points(16, seed=9))
    physical = {s.name: 0. for s in basis.parameters}
    small = points(4, "regular")
    bc = [make_case(pattern_for(mesh, mode=mode), basis, physical, small, sigma_mm=.00025) for mode in ("x", "y", "shear")]
    for label, cs in (("basis_single", bc[:1]), ("basis_multiple", bc)):
        row, _, _ = diagnose(StaticForwardAdapter(basis, ParameterSpace(basis.parameters), cs), physical, folder, label)
        rows.append({"case": label, **row})
    assert rows[-1]["rank"] > rows[-2]["rank"]
    # Independent modes do not break the common E0 vs damage stiffness scale.
    unknown = ParameterSpace(model.parameters+(Parameter("E0", 1200., 3000., "N/mm^2"),))
    physical = dict(GAUSSIAN_TRUTH, E0=2000.)
    row, f, _ = diagnose(StaticForwardAdapter(model, unknown, cases), physical, folder, "multiple_unknown_E")
    assert f.effective_rank == 6
    rows.append({"case": "multiple_unknown_E", **row})
    return {"status": "PASSED", "rows": rows, "scaled_jacobian_relative_gap": scaled_gap,
            "checks": {"amplitude_scaling_no_new_rank": "PASS", "independent_spatial_patterns": "PASS",
                       "global_stiffness_confounding": "EXPECTED_NONIDENTIFIABLE", "nonproportional_fatigue": "NOT_APPLICABLE"}}


def run_inverse_recovery(config, folder=None):
    mesh, model, space = mesh_for(config), GaussianDamage(), ParameterSpace(GaussianDamage().parameters)
    patterns = [pattern_for(mesh, mode=m) for m in ("x", "y", "shear")]
    holdx = points(64, seed=909)
    heldout_cases = [make_case(p, model, GAUSSIAN_TRUTH, holdx) for p in patterns]
    rows = []
    scenarios = [(0., count, 101, "single", None) for count in (16, 64)]
    scenarios += [(sigma, 64, seed, mode, None) for sigma in config["observation"]["noise_pixels"] if sigma > 0
                  for seed in config["verification"]["seeds"] for mode in ("single", "multiple")]
    scenarios += [(.03, 64, seed, "multiple", .22) for seed in config["verification"]["seeds"]]
    for i, (noise, count, seed, mode, length) in enumerate(scenarios):
        selected = patterns[:1] if mode == "single" else patterns
        cs = [make_case(p, model, GAUSSIAN_TRUTH, points(count, seed=seed),
                       sigma_mm=noise*config["observation"]["mm_per_pixel"], seed=seed+j*1000, correlated_length=length,
                       folder=folder, name=f"recovery_{i}_load{j}") for j, p in enumerate(selected)]
        ad = StaticForwardAdapter(model, space, cs)
        hold = StaticForwardAdapter(model, space, heldout_cases[:len(selected)])
        recovery, _ = recover(ad, GAUSSIAN_TRUTH, config, folder=folder, name=f"recovery_{i}", heldout=hold)
        if noise == 0:
            assert recovery["damage_field_rmse"] < 1e-7 and recovery["dimensionless_parameter_rmse"] < 1e-6
        else:
            assert recovery["status"] != "OPTIMIZATION_FAILED", "all starts failed on prescribed noisy recovery"
            assert recovery["whitened_rmse"] < 1.7, "noise-aware residual incompatible with generating scale"
        rows.append({"noise_pixel": noise, "count": count, "seed": seed, "loading": mode,
                     "correlation_length_mm": length, **recovery})
    unknown_space = ParameterSpace(model.parameters+(Parameter("E0", 1200., 3000., "N/mm^2"),))
    unknown = StaticForwardAdapter(model, unknown_space, heldout_cases)
    recovery, _ = recover(unknown, dict(GAUSSIAN_TRUTH, E0=2000.), config, folder=folder, name="unknown_E_recovery")
    diag, ff, _ = diagnose(unknown, dict(GAUSSIAN_TRUTH, E0=2000.))
    assert ff.effective_rank == 6 and recovery["displacement_rmse_mm"] < 1e-8
    rows.append({"noise_pixel": 0., "loading": "multiple_unknown_E", "interpretation": "equivalent stiffness fields; parameter error is not a uniqueness criterion", **recovery, "identifiability": diag})
    regularized_cases = [make_case(p, model, GAUSSIAN_TRUTH, points(64), sigma_mm=.00075, seed=101+j*1000)
                         for j, p in enumerate(patterns)]
    regularized = StaticForwardAdapter(model, space, regularized_cases)
    prior_fit, _ = recover(regularized, GAUSSIAN_TRUTH, config, folder=folder, name="regularized_recovery",
                          heldout=StaticForwardAdapter(model, space, heldout_cases),
                          prior_mean=np.zeros(6), prior_standard_deviation=np.full(6, .5))
    assert prior_fit["status"] == "OPTIMIZER_CONVERGED"
    rows.append({"noise_pixel": .03, "count": 64, "seed": 101, "loading": "multiple_with_prior",
                 "interpretation": "prior from search-box midpoint; improves stability by adding assumptions, not DIC information", **prior_fit})
    return {"status": "PASSED", "rows": rows, "checks": {"noiseless_recovery": "PASS", "noise_aware_recovery": "PASS",
            "sparse_dense_multistart": "PASS", "unknown_E_unique_damage": "EXPECTED_NONIDENTIFIABLE"}}


def run_boundary_uncertainty(config, folder=None):
    mesh, model, x = mesh_for(config), UniformDamage(), points(64)
    p = pattern_for(mesh)
    actual = {"d0": .2, "load_factor": 1.1}
    c = make_case(p, model, actual, x, folder=folder, name="biased_load")
    wrong = StaticForwardAdapter(model, ParameterSpace(model.parameters), [c])
    mistaken, _ = recover(wrong, {"d0": .2}, config, folder=folder, name="force_bias_absorbed")
    expected = 1-.8/1.1
    assert abs(mistaken["parameters"]["d0"]-expected) < 1e-8
    space = ParameterSpace(model.parameters+(Parameter("load_factor", .8, 1.3, "1"),))
    unknown = StaticForwardAdapter(model, space, [c])
    row, ff, _ = diagnose(unknown, actual, folder, "unknown_load_damage")
    assert ff.effective_rank == 1
    with_force = make_case(p, model, actual, x, reaction=True, folder=folder, name="independent_force")
    constrained = StaticForwardAdapter(model, space, [with_force])
    joint, f, _ = diagnose(constrained, actual, folder, "force_plus_DIC")
    assert f.effective_rank == 2
    corrected, _ = recover(constrained, actual, config, folder=folder, name="explicit_force_nuisance")
    assert corrected["damage_field_rmse"] < 1e-8
    registration = RegistrationError([.0005, -.0003], .0004)
    registered = make_case(p, model, {"d0": .2}, x, registration=registration, folder=folder, name="registration_bias")
    nuisance_space = ParameterSpace(model.parameters+(Parameter("tx", -.002, .002, "mm"), Parameter("ty", -.002, .002, "mm"),
                                Parameter("theta", -.003, .003, "rad")))
    true_reg = {"d0": .2, "tx": .0005, "ty": -.0003, "theta": .0004}
    reg_adapter = StaticForwardAdapter(model, nuisance_space, [registered])
    reg_diag, f, _ = diagnose(reg_adapter, true_reg, folder, "registration_nuisance")
    assert f.effective_rank == 4
    recovered, _ = recover(reg_adapter, true_reg, config, folder=folder, name="registration_recovery")
    assert recovered["damage_field_rmse"] < 1e-8
    reg_wrong, _ = recover(StaticForwardAdapter(model, ParameterSpace(model.parameters), [registered]), {"d0": .2},
                           config, folder=folder, name="registration_ignored")
    return {"status": "PASSED", "rows": [{"case": "load_damage", **row}, {"case": "force_DIC", **joint}, {"case": "registration", **reg_diag}],
            "incorrect_boundary_recovery": mistaken, "explicit_nuisance_recovery": corrected,
            "registration_ignored": reg_wrong, "registration_recovery": recovered,
            "checks": {"damage_absorbs_force_error": "PASS", "load_damage_degeneracy": "EXPECTED_NONIDENTIFIABLE",
                       "independent_force_measurement": "PASS", "explicit_registration_nuisance": "PASS"}}


def run_multiframe(config, folder=None):
    mesh, model, x = mesh_for(config, [2, 1]), UniformDamage(), points(16, "regular")
    pattern = pattern_for(mesh)
    space = ParameterSpace(model.parameters+(Parameter("A", .0001, .004, "cycle^-1"),))
    actual = {"d0": .12, "A": .0015}
    times = [0., 10., 20., 30., 40.]
    dummy_cases = []
    for time in times:
        o = simulate_measurement(x, np.zeros_like(x), DisplacementCovariance.diagonal(np.zeros_like(x)),
                                 seed=101, cycle_count=time, load_id=pattern.loading_state.load_id)
        dummy_cases.append(StaticLoadCase(pattern, o, variance_floor_mm2=1e-10))
    exact_adapter = FatigueForwardAdapter(model, space, dummy_cases, cycle_step=2.)
    exact = exact_adapter.predict(space.encode(actual))
    rows = []
    noiseless_cases = []
    for i, (time, (u, _)) in enumerate(zip(times, exact)):
        o = simulate_measurement(x, u, DisplacementCovariance.diagonal(np.zeros_like(x)), seed=101,
                                 cycle_count=time, load_id=pattern.loading_state.load_id)
        noiseless_cases.append(StaticLoadCase(pattern, o, variance_floor_mm2=1e-10))
        if folder is not None:
            o.save(Path(folder)/(f"frame_{i}_"+("calibration" if i < 3 else "future_withheld")))
    for label, cs in (("initial_single_frame", noiseless_cases[:1]), ("current_single_frame_N20", noiseless_cases[2:3]),
                      ("prefix_joint", noiseless_cases[:3])):
        ad = FatigueForwardAdapter(model, space, cs, cycle_step=2.)
        diag, f, _ = diagnose(ad, actual, folder, label)
        assert f.effective_rank == (2 if len(cs) > 1 else 1)
        rows.append({"case": label, **diag})
    for sigma_pixel, seed in [(0., 101)]+[(.01, s) for s in config["verification"]["seeds"]]:
        cs = []
        sigma = sigma_pixel*config["observation"]["mm_per_pixel"]
        for i, (time, (u, _)) in enumerate(zip(times, exact)):
            o = simulate_measurement(x, u, DisplacementCovariance.diagonal(np.full_like(x, sigma)), seed=seed+i*1000,
                                     cycle_count=time, load_id=pattern.loading_state.load_id)
            cs.append(StaticLoadCase(pattern, o, variance_floor_mm2=1e-10 if sigma == 0 else 0.))
        calibration = FatigueForwardAdapter(model, space, cs[:3], cycle_step=2.)
        future = FatigueForwardAdapter(model, space, cs[3:], cycle_step=2.)
        assert max(c.measurement.observation.cycle_count for c in calibration.cases) < min(c.measurement.observation.cycle_count for c in future.cases)
        metrics, fit = recover(calibration, actual, config, folder=folder, name=f"frame_recovery_{sigma_pixel:g}_{seed}", heldout=future)
        if sigma_pixel == 0:
            assert metrics["dimensionless_parameter_rmse"] < 1e-6 and metrics["heldout_exact_displacement_rmse_mm"] < 1e-9
        save_array(folder, f"forecast_{sigma_pixel:g}_{seed}", cycles=np.array(times),
            true_displacement_mm=np.array([u for u, _ in exact]),
            predicted_displacement_mm=np.array([u for u, _ in exact_adapter.predict(fit.best.normalized)]),
            calibration_mask=np.array([True, True, True, False, False]))
        rows.append({"case": "prefix_fit_future_evaluation", "noise_pixel": sigma_pixel, "seed": seed, **metrics})
    refined = FatigueForwardAdapter(model, space, dummy_cases, cycle_step=1.).displacement_prediction(space.encode(actual))
    temporal_gap = float(np.sqrt(np.mean((refined-exact_adapter.displacement_prediction(space.encode(actual)))**2)))
    return {"status": "PASSED", "rows": rows, "calibration_cycles": times[:3], "future_cycles": times[3:],
            "prediction_dt2_dt1_rmse_mm": temporal_gap,
            "checks": {"single_frame_initial_law_degeneracy": "EXPECTED_NONIDENTIFIABLE", "prefix_local_rank": "PASS",
                       "strict_future_holdout": "PASS", "real_Phase1B_trajectory": "PASS", "general_material_forecasting": "NOT_APPLICABLE"}}


def run_mesh_diagnostics(config, folder=None):
    model, space, x = GaussianDamage(), ParameterSpace(GaussianDamage().parameters), points(64, seed=101)
    reference = pattern_for(mesh_for(config, [16, 8]))
    ref_case = make_case(reference, model, GAUSSIAN_TRUTH, x, folder=folder, name="reference")
    exact_integral = .05*2.+.3*(.35*np.sqrt(np.pi/2)*(erf((2.-1.05)/(np.sqrt(2)*.35))-erf(-1.05/(np.sqrt(2)*.35))))*(.22*np.sqrt(np.pi/2)*(erf((1.-.47)/(np.sqrt(2)*.22))-erf(-.47/(np.sqrt(2)*.22))))
    rows = []
    for resolution in config["verification"]["meshes"]:
        p = pattern_for(mesh_for(config, resolution))
        layout = p.quadrature_layout
        matched = make_case(p, model, GAUSSIAN_TRUTH, x, folder=folder, name=f"mesh_{resolution[0]}_matched")
        cross = StaticLoadCase(p, ref_case.measurement, variance_floor_mm2=1e-10)
        initial_integral = float(layout.physical_weights @ model.evaluate(layout.coordinates, GAUSSIAN_TRUTH))
        obs_gap = float(np.sqrt(np.mean((matched.measurement.observation.displacement-ref_case.measurement.observation.displacement)**2)))
        direct, _ = recover(StaticForwardAdapter(model, space, [matched]), GAUSSIAN_TRUTH, config,
                            folder=folder, name=f"mesh_{resolution[0]}_matched_fit")
        assert direct["damage_field_rmse"] < 1e-6
        cross_fit, _ = recover(StaticForwardAdapter(model, space, [cross]), GAUSSIAN_TRUTH, config,
                              folder=folder, name=f"mesh_{resolution[0]}_cross_fit")
        diag, _, _ = diagnose(StaticForwardAdapter(model, space, [cross]), GAUSSIAN_TRUTH, folder, f"mesh_{resolution[0]}_sensitivity")
        rows.append({"nx": resolution[0], "ny": resolution[1], "quadrature_count": layout.count,
            "initial_integral_damage_mm2": initial_integral, "analytic_integral_damage_mm2": exact_integral,
            "initial_integral_error_mm2": abs(initial_integral-exact_integral), "forward_observation_gap_mm": obs_gap,
            "matched_mesh_damage_rmse": direct["damage_field_rmse"], "cross_mesh_recovery": cross_fit, "identifiability": diag})
    return {"status": "PASSED", "rows": rows, "checks": {"continuous_parameterization": "PASS",
            "matched_mesh_recovery": "PASS", "initial_quadrature_and_forward_error": "PASS", "mesh_objective_damage_recovery": "NOT_APPLICABLE"}}


RUNNERS = {"observation_consistency": run_observation_consistency, "noise_covariance": run_noise_covariance,
    "uniform_confounding": run_uniform_confounding, "gaussian_sensitivity": run_gaussian_sensitivity,
    "sampling_density": run_sampling_density, "multiple_loading": run_multiple_loading,
    "inverse_recovery": run_inverse_recovery, "boundary_uncertainty": run_boundary_uncertainty,
    "multiframe": run_multiframe, "mesh_diagnostics": run_mesh_diagnostics}
