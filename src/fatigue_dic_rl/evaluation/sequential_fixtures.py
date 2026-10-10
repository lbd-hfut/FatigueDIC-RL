"""Independent Phase 1B synthetic generation, kept out of estimator imports."""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.covariance import DisplacementCovariance
from fatigue_dic_rl.observations.noise import simulate_measurement
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase, ReactionObservation
from fatigue_dic_rl.inverse.parameterizations import UniformDamage, Parameter, ParameterSpace
from fatigue_dic_rl.fatigue import (MaterialHistoryState, FatigueEvolutionSolver, FatigueLoadCycle,
    FatigueStepRequest, LoadingMode, FatigueLawParameters, SyntheticFatigueDamageLaw)
from fatigue_dic_rl.inference.belief_state import ObservationFrame
from fatigue_dic_rl.inference.physics_predictor import PhysicsPredictor
from .identifiability_verification import mesh_for, pattern_for, points, material
from .inverse_metrics import SyntheticTruth


@dataclass(frozen=True)
class TruthFrame:
    cycle_count: float
    damage: np.ndarray
    displacement_dic_mm: np.ndarray
    response: object
    parameters: dict


def fatigue_loading(pattern, horizon=40.):
    return (FatigueLoadCycle(0., horizon, 1., -1., -1., ("left", "right"), LoadingMode.LOAD,
                             load_id=pattern.loading_state.load_id),)


def uniform_setup(config, *, fatigue=False, extra_parameters=(), fixed=None):
    mesh = mesh_for(config, [2, 1])
    pattern, model = pattern_for(mesh), UniformDamage()
    specs = model.parameters+((Parameter("A", .0001, .004, "cycle^-1"),) if fatigue else ())+tuple(extra_parameters)
    space = ParameterSpace(specs, fixed or {})
    predictor = PhysicsPredictor(model, space, pattern, load_history=fatigue_loading(pattern) if fatigue else ())
    x = points(config.get("observation", {}).get("count", 16), "random", seed=91)
    return model, space, pattern, predictor, x


def generate_truth(pattern, model, physical, cycles, coordinates, *, load_history=(), law_exponent=1.,
                   cycle_step=2., folder=None, name="truth"):
    """Direct solver rollout; does not call PhysicsPredictor or an inverse adapter."""
    factor = physical.get("load_factor", 1.)
    actual_pattern = replace(pattern,
        materials=pattern.materials if "E0" not in physical else material(physical["E0"], .3),
        traction_bcs=tuple(replace(b, values=b.values*factor) for b in pattern.traction_bcs),
        nodal_force_bcs=tuple(replace(b, values=b.values*factor) for b in pattern.nodal_force_bcs))
    layout = actual_pattern.quadrature_layout
    state = MaterialHistoryState.initialize(layout, layout.field(model.evaluate(layout.coordinates, physical), "1"),
                                            damage_max=model.damage_max)
    solver = None
    if load_history:
        law = SyntheticFatigueDamageLaw({int(r): FatigueLawParameters(physical["A"], .025, m=law_exponent)
                                        for r in np.unique(state.material_ids)})
        solver = FatigueEvolutionSolver(actual_pattern, state, law)
    mapping = ObservationOperator(pattern.mesh).prepare(coordinates, coordinate_units="mm", coordinate_system="specimen")
    result = []
    for index, target in enumerate(cycles):
        if solver is not None:
            while solver.committed_state.cycle_count < target:
                n = solver.committed_state.cycle_count
                segment = next(s for s in load_history if s.N_start <= n < s.N_end)
                end = min(target, segment.N_end, n+cycle_step)
                solver.advance(FatigueStepRequest(replace(segment, N_start=n, delta_N=end-n)))
            state = solver.committed_state
        response = Tri3MechanicsSolver().solve(replace(actual_pattern, damage=state.damage, cycle_count=target,
            loading_state=LoadingState(pattern.loading_state.load_id, "max" if load_history else "static")))
        u = mapping.apply(response.state.displacement).values
        truth = TruthFrame(target, state.damage.values, u, response, dict(physical))
        result.append(truth)
        if folder is not None:
            SyntheticTruth(layout.coordinates, truth.damage, physical, response.state.displacement.values,
                pattern.mesh.mesh_id, layout.reference.space_id, mesh_nodes_mm=pattern.mesh.nodes,
                strain_engineering=response.state.strain.values, stress_n_mm2=response.state.stress.values,
                reaction_n=response.reaction_forces.values).save(Path(folder)/f"{name}_{index}_truth")
    return tuple(result)


def observation_series(pattern, truths, coordinates, *, sigma_mm=5e-5, seed=100,
                       spatial=False, registration=None, reaction=False, missing=0., folder=None, name="frame"):
    if spatial:
        covariance = DisplacementCovariance.spatial(coordinates, sigma_mm**2*np.array([[1., .25], [.25, 1.6]]), .22)
    else:
        std = np.tile([sigma_mm, 1.25*sigma_mm], (len(coordinates), 1))
        covariance = DisplacementCovariance.diagonal(std)
    frames = []
    for i, t in enumerate(truths):
        m = simulate_measurement(coordinates, t.displacement_dic_mm, covariance, seed=seed*1009+i*104729,
            cycle_count=t.cycle_count, registration=registration, missing_fraction=missing,
            load_id=pattern.loading_state.load_id)
        r = None
        if reaction:
            ids = pattern.mesh.boundary_tags["left"].node_ids
            std = .01
            value = float(t.response.reaction_forces.values[ids, 0].sum())
            value += np.random.default_rng(seed*1031+i*105019).normal(0., std)
            r = ReactionObservation(ids, 0, value, std)
        frames.append(ObservationFrame(f"{name}-{i}", StaticLoadCase(pattern, m, r, 1e-10 if sigma_mm == 0 else 0.)))
        if folder is not None:
            m.save(Path(folder)/f"{name}_{i}_observations")
    return tuple(frames)
