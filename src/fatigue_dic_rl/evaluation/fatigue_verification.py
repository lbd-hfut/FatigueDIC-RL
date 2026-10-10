"""Phase 1B scientific experiments: real FE coupling and independent ODE references."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.fatigue import (ConstantFatigueHazardLaw, FatigueEvolutionSolver, FatigueLawParameters,
    FatigueLoadCycle, FatigueSolverConfig, FatigueStepError, FatigueStepRequest, LoadingMode,
    MaterialHistoryState, SyntheticFatigueDamageLaw, run_trajectory)
from fatigue_dic_rl.mechanics import MechanicsConvergenceError, SolverDiagnostics
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverConfig
from fatigue_dic_rl.observations import ObservationOperator
from .tri3_meshes import rectangle_mesh
from .tri3_verification import tension_request


def make_solver(config, *, control="load", nx=None, ny=None, localized=False, mechanics=None):
    geo, mat = config["geometry"], config["material"]
    mesh = rectangle_mesh(geo["nx"] if nx is None else nx, geo["ny"] if ny is None else ny,
        width_mm=geo["width_mm"], height_mm=geo["height_mm"], seed=geo["seed"], jitter=geo["jitter"])
    pattern = tension_request(mesh, mat["young_modulus_n_per_mm2"], mat["poisson_ratio"],
        geo["thickness_mm"], config["loading"]["traction_n_per_mm2"], control=control,
        extension_mm=config["loading"]["extension_mm"])
    layout = pattern.quadrature_layout
    damage = np.full(layout.count, config["initial"]["damage"])
    if localized:
        # Same smooth physical-coordinate imperfection for every mesh, not a prescribed final hotspot.
        x, y = layout.coordinates.T
        initial = config["initial"]
        damage += initial["imperfection_amplitude"]*np.exp(-.5*(
            ((x-geo["width_mm"]/2)/initial["sigma_x_mm"])**2+
            ((y-geo["height_mm"]/2)/initial["sigma_y_mm"])**2))
    state = MaterialHistoryState.initialize(layout, layout.field(damage, "1"), damage_max=config["initial"]["damage_max"])
    fractions = np.array([[.13, .23], [.43, .71], [.81, .37], [.61, .83], [1.2, .4]])
    mapping = ObservationOperator(mesh).prepare(fractions*[geo["width_mm"], geo["height_mm"]],
        coordinate_units="mm", coordinate_system="specimen")
    return FatigueEvolutionSolver(pattern, state, SyntheticFatigueDamageLaw({0: FatigueLawParameters(**config["fatigue_law"])}),
        mechanics=Tri3MechanicsSolver(Tri3SolverConfig(**config.get("mechanics", {}))) if mechanics is None else mechanics,
        config=FatigueSolverConfig(**config["evolution"]), observation_mapping=mapping)


def cycles(config, *, dt=None, control="load", R=None, total=None):
    settings = config["loading"]
    dt = settings["delta_N"] if dt is None else dt
    total = settings["total_N"] if total is None else total
    R = settings["R"] if R is None else R
    if not np.isclose(total/dt, round(total/dt), rtol=0, atol=1e-10):
        raise ContractError("verification increments must partition total_N exactly")
    mode = LoadingMode.LOAD if control == "load" else LoadingMode.DISPLACEMENT
    return tuple(FatigueLoadCycle(i*dt, dt, 1, R, R, ("left", "right"), mode) for i in range(round(total/dt)))


def simulate(config, folder=None, *, name="trajectory", control="load", dt=None, nx=None, ny=None, localized=False,
             R=None, law=None, total=None):
    solver = make_solver(config, control=control, nx=nx, ny=ny, localized=localized)
    if law is not None:
        solver.damage_law = law
    trajectory = run_trajectory(solver, cycles(config, control=control, dt=dt, R=R, total=total),
        random_seed=config["geometry"]["seed"], code_version=config.get("code_version", "verification-in-process"))
    if folder is not None:
        trajectory.save(folder/name)
    return trajectory


def metrics(trajectory):
    layout = trajectory.initial_state.layout
    right = layout.mesh.boundary_tags["right"].node_ids
    result = []
    history = (trajectory.initial_state, *(r.state_after for r in trajectory.substeps))
    responses = (trajectory.initial_maximum, *(r.maximum for r in trajectory.substeps))
    cumulative = 0
    for i, (state, response) in enumerate(zip(history, responses, strict=True)):
        row = trajectory.substeps[i-1] if i else None
        if row:
            cumulative += row.diagnostics["max_damage_dissipation_estimate_n_mm"]
        result.append({"N_cycle": state.cycle_count,
            "damage_mean": float(np.average(state.damage.values, weights=layout.physical_weights)),
            "damage_max": float(state.damage.values.max()),
            "right_u_mm": float(response.state.displacement.values[right, 0].mean()),
            "right_reaction_n": float(response.reaction_forces.values[right, 0].sum()),
            "left_reaction_n": float(response.reaction_forces.values[layout.mesh.boundary_tags["left"].node_ids, 0].sum()),
            "elastic_energy_n_mm": response.elastic_strain_energy,
            "cumulative_max_phase_damage_dissipation_estimate_n_mm": cumulative,
            "residual_n": response.diagnostics.residual_norm})
    return result


def assert_physical(trajectory):
    previous = trajectory.initial_state
    for row in trajectory.substeps:
        assert row.state_after.cycle_count == row.cycle.N_end
        assert np.all(row.state_after.damage.values >= previous.damage.values)
        assert np.all(row.state_after.damage.values < previous.damage_max)
        np.testing.assert_allclose(row.state_after.accumulated_damage_increment.values,
            row.state_after.damage.values-trajectory.initial_state.damage.values, rtol=0, atol=2e-15)
        assert row.diagnostics["max_damage_dissipation_estimate_n_mm"] >= 0
        assert row.diagnostics["min_damage_dissipation_estimate_n_mm"] >= 0
        previous = row.state_after


def run_constant(config, folder=None):
    hazard = config["verification"]["hazard_per_cycle"]
    trajectories = [simulate(config, folder, name=f"split_{factor}", dt=config["loading"]["delta_N"]/factor,
                             law=ConstantFatigueHazardLaw(hazard)) for factor in (1, 2, 4, 8)]
    h = trajectories[0].initial_state
    exact = h.damage_max-(h.damage_max-h.damage.values)*np.exp(-hazard*config["loading"]["total_N"])
    errors = []
    for t in trajectories:
        assert_physical(t)
        error = float(np.max(np.abs(t.substeps[-1].state_after.damage.values-exact)))
        assert error < 4e-15
        errors.append(error)
    zero = simulate(config, folder, name="zero_hazard", law=ConstantFatigueHazardLaw(0))
    np.testing.assert_array_equal(zero.substeps[-1].state_after.damage.values, zero.initial_state.damage.values)
    return {"status": "PASS", "analytic_max_errors": errors, "subdivision_factors": [1, 2, 4, 8],
            "final_damage": float(exact[0]), "zero_hazard_invariant": True, "rows": metrics(trajectories[0])}


def run_reversed(config, folder=None):
    reversed_t = simulate(config, folder, name="fully_reversed", R=-1)
    zero_t = simulate(config, folder, name="zero_amplitude", R=1)
    driving = reversed_t.initial_driving
    np.testing.assert_allclose(driving.Y_max.values, driving.Y_min.values, atol=1e-14, rtol=1e-13)
    assert driving.energy_amplitude.values.min() > 0
    assert np.all(reversed_t.substeps[-1].state_after.damage.values > reversed_t.initial_state.damage.values)
    np.testing.assert_array_equal(zero_t.initial_driving.strain_amplitude.values, 0)
    np.testing.assert_array_equal(zero_t.initial_driving.energy_amplitude.values, 0)
    np.testing.assert_array_equal(zero_t.substeps[-1].state_after.damage.values, zero_t.initial_state.damage.values)
    return {"status": "PASS", "Ya_reversed_min_n_mm2": float(driving.energy_amplitude.values.min()),
            "endpoint_energy_difference_max_n_mm2": float(np.max(np.abs(driving.Y_max.values-driving.Y_min.values))),
            "zero_amplitude_invariant": True, "rows": metrics(reversed_t)}


def uniform_exact(config, N, control):
    """Independent analytic integral for m=1, eta=beta=0, homogeneous uniaxial response."""
    p, g, mat, loading = config["fatigue_law"], config["geometry"], config["material"], config["loading"]
    assert p["m"] == 1 and p["eta"] == 0 and p["beta_R"] == 0
    E, D0, cap = mat["young_modulus_n_per_mm2"], config["initial"]["damage"], config["initial"]["damage_max"]
    amplitude_factor = (1-loading["R"])/2
    if control == "displacement":
        Ya = .5*E*(amplitude_factor*loading["extension_mm"]/g["width_mm"])**2
        return cap-(cap-D0)*np.exp(-N*p["A_per_cycle"]*Ya/p["Y_ref_n_per_mm2"])
    k = p["A_per_cycle"]*.5*(loading["traction_n_per_mm2"]*amplitude_factor)**2/E/p["Y_ref_n_per_mm2"]
    if N == 0 or k == 0:
        return D0
    a, s0 = 1-cap, cap-D0

    def equation(D):
        s = cap-D
        return (a*a*np.log(s0/s)+2*a*(s0-s)+(s0*s0-s*s)/2)/k-N

    return brentq(equation, D0, np.nextafter(cap, 0), xtol=1e-14)


def run_uniform(config, folder=None):
    rows, errors = [], []
    E, width, height, thickness = (config["material"]["young_modulus_n_per_mm2"],
        config["geometry"]["width_mm"], config["geometry"]["height_mm"], config["geometry"]["thickness_mm"])
    sigma = config["loading"]["traction_n_per_mm2"]
    for control in ("load", "displacement"):
        for factor in (1, 2, 4, 8):
            t = simulate(config, folder, name=f"{control}_{factor}", control=control,
                         dt=config["loading"]["delta_N"]/factor)
            assert_physical(t)
            final = t.substeps[-1]
            D = float(final.state_after.damage.values.mean())
            np.testing.assert_allclose(final.state_after.damage.values, D, rtol=0, atol=2e-13)
            m = metrics(t)[-1]
            if control == "load":
                np.testing.assert_allclose(m["right_u_mm"], width*sigma/(E*(1-D)), rtol=2e-11)
                np.testing.assert_allclose(m["left_reaction_n"], -sigma*height*thickness, rtol=2e-11)
                assert m["elastic_energy_n_mm"] > t.initial_maximum.elastic_strain_energy
            else:
                extension = config["loading"]["extension_mm"]
                np.testing.assert_allclose(m["right_u_mm"], extension, rtol=1e-13)
                np.testing.assert_allclose(m["right_reaction_n"], E*(1-D)*extension/width*height*thickness, rtol=2e-11)
                assert m["elastic_energy_n_mm"] < t.initial_maximum.elastic_strain_energy
            exact = uniform_exact(config, config["loading"]["total_N"], control)
            error = abs(D-exact)
            if control == "displacement":
                assert error < 2e-13  # Uniform imposed strain gives genuinely constant hazard.
            else:
                errors.append(error)
            rows.append(dict(m, control=control, delta_N=config["loading"]["delta_N"]/factor,
                             effective_E_n_mm2=E*(1-D), analytic_final_damage=exact, damage_abs_error=error))
    assert all(a > b for a, b in zip(errors, errors[1:]))
    rates = [float(np.log2(a/b)) for a, b in zip(errors, errors[1:])]
    assert rates[-1] > .9
    return {"status": "PASS", "rows": rows, "load_temporal_orders": rates,
            "analytic_derivation": "k*N=a^2*ln(s0/s)+2a*(s0-s)+(s0^2-s^2)/2; s=Dmax-D, a=1-Dmax"}


def run_localized(config, folder=None):
    t = simulate(config, folder, localized=True)
    assert_physical(t)
    initial, final = t.initial_state, t.substeps[-1]
    increment = final.state_after.damage.values-initial.damage.values
    Ya0 = t.initial_driving.energy_amplitude.values
    assert np.ptp(Ya0) > .1*Ya0.mean()
    assert np.ptp(increment) > .001
    hotspot = int(np.argmax(increment))
    weak = initial.damage.values > np.quantile(initial.damage.values, .7)
    far = initial.damage.values < np.quantile(initial.damage.values, .3)
    assert increment[weak].mean() > increment[far].mean()
    strain_change = float(np.linalg.norm(final.driving_end.strain_amplitude.values-t.initial_driving.strain_amplitude.values))
    stress_change = float(np.linalg.norm(final.maximum.state.stress.values-t.initial_maximum.state.stress.values))
    assert strain_change > 1e-6 and stress_change > 1e-4
    return {"status": "PASS", "rows": metrics(t), "hotspot_coordinate_mm": initial.layout.coordinates[hotspot].tolist(),
            "Ya_initial_min_n_mm2": float(Ya0.min()), "Ya_initial_max_n_mm2": float(Ya0.max()),
            "max_evolved_damage_increment": float(increment.max()),
            "weak_region_mean_increment": float(increment[weak].mean()),
            "far_region_mean_increment": float(increment[far].mean()),
            "strain_amplitude_change_array_l2": strain_change, "stress_redistribution_array_l2_n_mm2": stress_change,
            "damage_is_crack_geometry": False}


def continuous_reference(config, *, rtol=1e-10, atol=1e-12):
    """Independent DOP853 integration of the FE semi-discrete ODE, including Ymax*dD/dN.

    Shares the defined mechanics/constitutive model; does not use DamageTransition
    or advance(). This tests fatigue time error on a fixed spatial discretization.
    """
    solver = make_solver(config, localized=True)
    initial = solver.committed_state
    layout = initial.layout
    cycle = cycles(config, dt=config["loading"]["total_N"])[0]
    weights = layout.physical_weights*solver.pattern.thickness_mm

    def rhs(N, vector):
        state = MaterialHistoryState.initialize(layout, layout.field(vector[:-1], "1"),
            damage_max=initial.damage_max, cycle_count=N)
        minimum, maximum, driving = solver._endpoints(state, cycle, N)
        hazard = solver.damage_law.hazard(driving, cycle, state)
        rate = (initial.damage_max-vector[:-1])*hazard.values
        return np.r_[rate, np.dot(weights*driving.Y_max.values, rate)]

    ode = solve_ivp(rhs, (0, cycle.N_end), np.r_[initial.damage.values, 0.], method="DOP853", rtol=rtol, atol=atol)
    assert ode.success
    state = MaterialHistoryState.initialize(layout, layout.field(ode.y[:-1, -1], "1"),
        damage_max=initial.damage_max, cycle_count=cycle.N_end)
    _, endpoint, driving = solver._endpoints(state, cycle, cycle.N_end)
    return state, endpoint, driving, float(ode.y[-1, -1]), ode.nfev


def damage_rms(layout, diff):
    return float(np.sqrt(np.average(np.sum(diff**2, axis=1) if diff.ndim == 2 else diff**2,
                                    weights=layout.physical_weights)))


def displacement_rms(mesh, values):
    # Exact integral of the squared P1 field, independently using the triangle mass matrix.
    total, area = 0., 0.
    from fatigue_dic_rl.geometry.triangles import Tri3Geometry
    for cell in mesh.connectivity:
        A = Tri3Geometry.from_vertices(mesh.nodes[cell]).area
        v = values[cell]
        total += A/12*(np.sum(v*v)+np.sum(v.sum(axis=0)**2))
        area += A
    return float(np.sqrt(total/area))


def run_temporal(config, folder=None):
    ref, response, driving, W, nfev = continuous_reference(config)
    ref_fine, _, _, W_fine, fine_nfev = continuous_reference(config, rtol=2e-12, atol=2e-14)
    reference_difference = damage_rms(ref.layout, ref.damage.values-ref_fine.damage.values)
    assert reference_difference < 1e-8 and abs(W-W_fine) < 1e-9
    if folder is not None:
        np.savez_compressed(folder/"ode_reference.npz", damage=ref.damage.values,
            displacement_mm=response.state.displacement.values, strain_amplitude=driving.strain_amplitude.values,
            stress_n_mm2=response.state.stress.values, cumulative_dissipation_n_mm=W,
            tighter_reference_damage=ref_fine.damage.values)
    rows = []
    for factor in (1, 2, 4, 8, 32):
        dt = config["loading"]["delta_N"]/factor
        t = simulate(config, folder, name=f"dt_factor_{factor}", dt=dt, localized=True)
        assert not any(s.rejected_attempts for s in t.steps), "step-size study must use specified fixed increments"
        assert_physical(t)
        final = t.substeps[-1]
        diff = final.state_after.damage.values-ref.damage.values
        stress_diff = (final.maximum.state.stress.values-response.state.stress.values)*np.sqrt([1, 1, 2])
        strain_diff = (final.driving_end.strain_amplitude.values-driving.strain_amplitude.values)*np.sqrt([1, 1, .5])
        row = {"delta_N_cycle": dt, "damage_rms_error": damage_rms(ref.layout, diff),
               "damage_max_abs_error": abs(float(final.state_after.damage.values.max()-ref.damage.values.max())),
               "u_rms_error_mm": displacement_rms(ref.layout.mesh,
                   final.maximum.state.displacement.values-response.state.displacement.values),
               "stress_tensor_rms_error_n_mm2": damage_rms(ref.layout, stress_diff),
               "strain_amplitude_tensor_rms_error": damage_rms(ref.layout, strain_diff),
               "elastic_energy_abs_error_n_mm": abs(final.maximum.elastic_strain_energy-response.elastic_strain_energy),
               "cumulative_dissipation_abs_error_n_mm": abs(metrics(t)[-1][
                   "cumulative_max_phase_damage_dissipation_estimate_n_mm"]-W),
               "final_damage_max": float(final.state_after.damage.values.max())}
        if rows:
            row["damage_order"] = float(np.log(rows[-1]["damage_rms_error"]/row["damage_rms_error"])/
                                        np.log(rows[-1]["delta_N_cycle"]/dt))
            for key in ("damage_rms_error", "u_rms_error_mm", "stress_tensor_rms_error_n_mm2",
                        "elastic_energy_abs_error_n_mm", "cumulative_dissipation_abs_error_n_mm"):
                assert row[key] < rows[-1][key]
        rows.append(row)
    assert rows[3]["damage_order"] > .9 and rows[-1]["damage_order"] > .9
    return {"status": "PASS", "rows": rows, "reference": "DOP853 FE semi-discrete ODE; no hazard freezing",
            "reference_nfev": nfev, "tighter_reference_nfev": fine_nfev,
            "reference_damage_rms_difference": reference_difference,
            "reference_dissipation_difference_n_mm": abs(W-W_fine),
            "reference_cumulative_dissipation_n_mm": W, "spatial_error_validated_by_this_test": False}


def run_mesh(config, folder=None):
    rows, histories, sampled_u = [], [], []
    for n in config["verification"]["mesh_nx"]:
        t = simulate(config, folder, name=f"mesh_{n}", nx=n, ny=n//2, localized=True)
        fine_t = simulate(config, folder, name=f"mesh_{n}_half_dt", nx=n, ny=n//2, localized=True,
                          dt=config["loading"]["delta_N"]/2)
        assert_physical(t)
        assert_physical(fine_t)
        final = t.substeps[-1]
        layout = t.initial_state.layout
        u = final.observation.displacement[:4]
        sampled_u.append(u)
        mesh = layout.mesh
        h = max(np.linalg.norm(np.roll(mesh.nodes[c], -1, axis=0)-mesh.nodes[c], axis=1).max() for c in mesh.connectivity)
        m = metrics(t)[-1]
        maximum = int(np.argmax(final.state_after.damage.values))
        rows.append(dict(m, nx=n, V=mesh.n_nodes, E=mesh.n_elements, h_max_mm=float(h),
            initial_damage_max=float(t.initial_state.damage.values.max()),
            initial_damage_mean=float(np.average(t.initial_state.damage.values, weights=layout.physical_weights)),
            max_evolved_increment=float((final.state_after.damage.values-t.initial_state.damage.values).max()),
            temporal_half_dt_damage_max_change=float(fine_t.substeps[-1].state_after.damage.values.max()-m["damage_max"]),
            temporal_half_dt_right_u_change_mm=metrics(fine_t)[-1]["right_u_mm"]-m["right_u_mm"],
            hotspot_x_mm=float(layout.coordinates[maximum, 0]), hotspot_y_mm=float(layout.coordinates[maximum, 1]),
            area_above_damage_threshold_mm2=float(layout.physical_weights[
                final.state_after.damage.values > config["verification"]["damage_threshold"]].sum())))
        histories.append({"nx": n, "rows": metrics(t)})
    for i in range(1, len(rows)):
        rows[i]["damage_max_change_from_previous_mesh"] = rows[i]["damage_max"]-rows[i-1]["damage_max"]
        rows[i]["common_DIC_u_rms_change_mm"] = float(np.sqrt(np.mean((sampled_u[i]-sampled_u[i-1])**2)))
    spread = max(r["damage_max"] for r in rows)-min(r["damage_max"] for r in rows)
    return {"status": "PASS", "rows": rows, "histories": histories, "damage_max_range": spread,
            "mesh_objective_localization_verified": False,
            "interpretation": "local softening model; mesh sensitivity measured, no spatial convergence or crack geometry claim"}


def run_rollback(config, folder=None):
    examples = []
    for reason in ("large_step", "invalid_material", "underconstrained", "nonfinite_hazard", "late_substep_failure"):
        solver = make_solver(config)
        cycle = cycles(config)[0]
        if reason == "large_step":
            solver.config = replace(solver.config, max_delta_damage_per_step=1e-5, adaptive_subdivision=False)
        elif reason == "invalid_material":
            assignment = replace(solver.pattern.materials[0],
                parameters={"young_modulus_n_per_mm2": -1, "poisson_ratio": .3})
            solver.pattern = replace(solver.pattern, materials={0: assignment})
        elif reason == "underconstrained":
            solver.pattern = replace(solver.pattern, displacement_bcs=())
        elif reason == "nonfinite_hazard":
            solver.damage_law = SyntheticFatigueDamageLaw({0: FatigueLawParameters(.002, .025, beta_R=-1e5)})
        else:
            class FailLate(Tri3MechanicsSolver):
                def _solve(self, request):
                    if request.cycle_count >= cycle.delta_N*.75 and request.loading_state.phase == "max":
                        raise MechanicsConvergenceError(SolverDiagnostics(False, "injected-failure", 1, 1, "late trial failure"))
                    return super()._solve(request)
            solver.mechanics = FailLate()
            solver.config = replace(solver.config, max_delta_damage_per_step=.01)
        original = solver.committed_state
        try:
            solver.advance(FatigueStepRequest(cycle))
        except FatigueStepError as exc:
            assert solver.committed_state is original
            assert solver.committed_state.cycle_count == 0
            examples.append({"case": reason, "status": "EXPECTED_FAILURE", "history_unchanged": True,
                             "partial_checkpoint_committed": False, "diagnostics": [asdict(d) for d in exc.failures]})
        else:
            raise AssertionError(f"failure case {reason} silently passed")
    return {"status": "PASS", "expected_failures": examples}


def run_synthetic(config, folder=None):
    first = simulate(config, folder, name="trajectory", localized=True)
    second = simulate(config, folder, name="repeat_trajectory", localized=True)
    assert_physical(first)
    for a, b in zip(first.substeps, second.substeps, strict=True):
        for x, y in ((a.state_after.damage.values, b.state_after.damage.values),
                     (a.maximum.state.displacement.values, b.maximum.state.displacement.values),
                     (a.maximum.state.stress.values, b.maximum.state.stress.values),
                     (a.hazard_used.values, b.hazard_used.values),
                     (a.observation.displacement, b.observation.displacement)):
            np.testing.assert_array_equal(x, y)
    assert all(o.valid_mask.tolist() == [True, True, True, True, False] for o in first.observations)
    return {"status": "PASS", "rows": metrics(first), "bitwise_reproducible_numerical_arrays": True,
            "frames_including_initial": len(first.observations), "truth_observation_interfaces_separate": True,
            "image_correlation_performed": False, "random_seed": config["geometry"]["seed"]}


RUNNERS = {"constant_hazard": run_constant, "reversed_loading": run_reversed, "uniform_damage": run_uniform,
           "localized_damage": run_localized, "temporal_convergence": run_temporal,
           "mesh_sensitivity": run_mesh, "rollback": run_rollback, "synthetic_trajectory": run_synthetic}
