"""Deterministic analytic-field and physical-update smoke; this is not an FE solve."""

from dataclasses import replace
from typing import Any

import numpy as np

from .core import CycleLoading, FieldData, FieldLocation, LoadingState, MechanicsState, StateRole
from .geometry import BoundaryTag, MeshData
from .materials import ConstantHazardLaw
from .observations import DICObservation, ObservationOperator
from .physics import DamageTransition


def run_smoke(*, dmax: float = 0.95, hazard_per_cycle: float = 0.002,
              cycle_steps: tuple[float, ...] = (0, 1, 10, 100, 1000)) -> dict[str, Any]:
    mesh = MeshData(nodes=[[0, 0], [2, 0], [1.7, 1.2], [-0.2, 1]],
                    connectivity=[[0, 1, 2], [0, 2, 3]], element_types=["tri3", "tri3"],
                    material_region_ids=[0, 1], coordinate_units="mm", coordinate_system="specimen",
                    boundary_tags={"bottom": BoundaryTag([0, 1], [[0, 1]])}, mesh_id="phase0-smoke-mesh")
    coefficients = np.array([[0.3, 0.012, -0.021], [-0.4, 0.008, 0.015]])

    def affine(points: np.ndarray) -> np.ndarray:
        return np.column_stack((np.ones(len(points)), points)) @ coefficients.T

    displacement = FieldData(affine(mesh.nodes), FieldLocation.NODE, "mm",
                             reference=mesh.field_reference(FieldLocation.NODE))
    points = np.array([[0.23, 0.12], [1.4, 0.26], [0.51, 0.78], [0.85, 0.6],
                       [0, 0], [-0.08, 0.7], [3, 0.5], [-0.5, -0.3]])
    mapping = ObservationOperator(mesh).prepare(points, coordinate_units="mm", coordinate_system="specimen")
    prediction = mapping.apply(displacement)
    expected_mask = np.array([True]*6 + [False]*2)
    np.testing.assert_array_equal(prediction.valid_mask, expected_mask)
    error = float(np.max(np.abs(prediction.values[expected_mask] - affine(points[expected_mask]))))
    np.testing.assert_allclose(prediction.values[expected_mask], affine(points[expected_mask]), rtol=0,
                               atol=16*np.finfo(float).eps)
    if not np.all(np.isnan(prediction.values[~expected_mask])):
        raise AssertionError("outside points must have NaN predictions")
    # Construct a measurement-shaped record; no damage truth is stored in it.
    observation = DICObservation(points, prediction.values, prediction.valid_mask,
                                 expected_mask.astype(float), 0, "peak", "smoke-load", "specimen", "mm", "mm")
    initial = FieldData([0.0, 0.2], FieldLocation.ELEMENT, "1",
                        reference=mesh.field_reference(FieldLocation.ELEMENT))
    loading = LoadingState("smoke-load", "peak")
    state = MechanicsState(mesh.mesh_id, StateRole.PREDICTION, initial, 0, loading)
    law = ConstantHazardLaw()
    transition = DamageTransition(dmax)
    history = [state.damage.values.copy()]
    for step in cycle_steps:
        interval = CycleLoading(step, loading)
        hazard = law.hazard(state, {"hazard_per_cycle": hazard_per_cycle}, interval)
        updated = transition.apply(state.damage, hazard, interval.delta_cycles)
        state = replace(state, damage=updated, cycle_count=state.cycle_count + interval.delta_cycles)
        history.append(updated.values.copy())
    trajectory = np.array(history)
    if not (np.all(np.diff(trajectory, axis=0) >= 0) and np.all(trajectory >= 0) and np.all(trajectory <= dmax)):
        raise AssertionError("damage bounds or irreversibility failed")
    expected_damage = dmax - (dmax - initial.values) * np.exp(-sum(cycle_steps) * hazard_per_cycle)
    damage_error = float(np.max(np.abs(state.damage.values - expected_damage)))
    np.testing.assert_allclose(state.damage.values, expected_damage, rtol=0, atol=16*np.finfo(float).eps)
    return {"test_kind": "analytic_field_and_module_interfaces", "fe_solve_performed": False,
            "V": mesh.n_nodes, "E": mesh.n_elements, "Q": len(observation.coordinates),
            "valid_observation_points": int(expected_mask.sum()), "outside_points": int((~expected_mask).sum()),
            "affine_max_abs_error_mm": error, "damage_max_abs_error": damage_error,
            "total_cycles": state.cycle_count, "dmax": dmax, "final_damage": state.damage.values.tolist(),
            "irreversibility_passed": True, "damage_bounds_passed": True}
