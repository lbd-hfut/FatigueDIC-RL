from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import FieldData, LoadingState
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import BoundaryTag, MeshData, QuadratureLayout
from fatigue_dic_rl.materials import IsotropicPlaneStress
from fatigue_dic_rl.mechanics import (DisplacementBC, MaterialAssignment, MechanicsConvergenceError,
                                     NodalForceBC, SolverRequest, TractionBC)
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.mechanics.tri3_contracts import BodyForceDensity, Tri3SolverConfig, Tri3SolverRequest
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.tri3_verification import material, tension_request


@pytest.fixture
def elastic_request():
    mesh = rectangle_mesh(3, 2, width_mm=3, height_mm=2, seed=12)
    return tension_request(mesh, 70000, 0.3, 0.8, 10)


def test_response_contract_residual_and_no_evolution(elastic_request):
    result = Tri3MechanicsSolver().solve(elastic_request)
    assert result.state.displacement.units == "mm" and result.state.displacement.location.value == "node"
    assert result.state.strain.units == "1" and result.state.strain.location.value == "quadrature"
    assert result.state.stress.units == "N/mm^2" and result.state.stress.location.value == "quadrature"
    assert result.reaction_forces.units == "N" and result.energy_units == "N*mm"
    assert result.diagnostics.converged and result.diagnostics.residual_norm < 1e-9
    np.testing.assert_array_equal(result.state.damage.values, elastic_request.damage.values)
    assert result.state.cycle_count == elastic_request.cycle_count
    assert result.assembly.thickness_mm == 0.8


def test_legacy_pa_material_traction_and_meter_thickness(elastic_request):
    legacy = SolverRequest(elastic_request.mesh, elastic_request.damage,
        {0: MaterialAssignment(IsotropicPlaneStress(), {"young_modulus_pa": 70e9, "poisson_ratio": 0.3})},
        elastic_request.loading_state, 0, displacement_bcs=elastic_request.displacement_bcs,
        traction_bcs=(TractionBC("right", [1e7, 0], "Pa"),), thickness_m=0.0008)
    solver = Tri3MechanicsSolver()
    np.testing.assert_allclose(solver.solve(legacy).state.displacement.values,
                               solver.solve(elastic_request).state.displacement.values, rtol=1e-13, atol=1e-16)


def test_optional_damage_and_explicit_element_lift(elastic_request):
    zero = replace(elastic_request, damage=None)
    assert np.all(zero.damage.values == 0)
    element = FieldData(np.full(zero.mesh.n_elements, 0.2), "element", "1", reference=zero.mesh.field_reference("element"))
    result = Tri3MechanicsSolver().solve(replace(zero, damage=element))
    assert result.state.damage is element
    assert result.assembly.quadrature_damage.location.value == "quadrature"
    np.testing.assert_array_equal(result.assembly.quadrature_damage.values, element.values)


def test_node_damage_is_not_implicitly_projected(elastic_request):
    field = FieldData(np.zeros(elastic_request.mesh.n_nodes), "node", "1", reference=elastic_request.mesh.field_reference("node"))
    with pytest.raises(ContractError, match="node projection"):
        Tri3MechanicsSolver().solve(replace(elastic_request, damage=field))


@pytest.mark.parametrize("updates", [{"damage_max": 1}, {"damage_max": -0.1}, {"residual_atol_n": -1},
                                     {"load_quadrature_order": 0}, {"load_quadrature_order": 1}, {"minimum_pivot_ratio": 1}])
def test_illegal_solver_configuration(updates):
    with pytest.raises(ContractError):
        Tri3SolverConfig(**updates)


def test_damage_cap_and_bad_material_rejected(elastic_request):
    with pytest.raises(ContractError, match="damage_max"):
        Tri3MechanicsSolver().solve(replace(elastic_request, damage=elastic_request.damage.with_values(np.ones(elastic_request.damage.count))))
    bad = material(-1, 0.3)
    with pytest.raises(ContractError):
        Tri3MechanicsSolver().solve(replace(elastic_request, materials=bad))


def test_underconstrained_rigid_modes(elastic_request):
    solver = Tri3MechanicsSolver()
    for bcs in [(), (DisplacementBC([0, 0], [0, 1], [0, 0], "mm"),)]:
        with pytest.raises(MechanicsConvergenceError, match="rigid-body") as exc:
            solver.solve(replace(elastic_request, displacement_bcs=bcs))
        assert exc.value.diagnostics.converged is False


def test_duplicate_constraints_and_loads(elastic_request):
    solver = Tri3MechanicsSolver()
    with pytest.raises(ContractError, match="duplicate prescribed"):
        solver.solve(replace(elastic_request, displacement_bcs=elastic_request.displacement_bcs+(elastic_request.displacement_bcs[0],)))
    with pytest.raises(ContractError, match="conflicting"):
        replace(elastic_request, displacement_bcs=(DisplacementBC([0, 0], [0, 0], [0, 1], "mm"),))
    with pytest.raises(ContractError, match="duplicate nodal"):
        solver.solve(replace(elastic_request, nodal_force_bcs=(NodalForceBC([2, 2], [[1, 0], [1, 0]]),)))
    with pytest.raises(ContractError, match="traction facets"):
        solver.solve(replace(elastic_request, traction_bcs=elastic_request.traction_bcs*2))


def test_consistent_traction_and_body_force_thickness(elastic_request):
    solver = Tri3MechanicsSolver()
    base = solver.assemble(elastic_request)
    np.testing.assert_allclose(base.force.reshape((-1, 2)).sum(axis=0), [10*2*0.8, 0], atol=1e-13)
    request = replace(elastic_request, traction_bcs=(), body_force=BodyForceDensity([1.2, -0.8]))
    force = solver.assemble(request).force.reshape((-1, 2)).sum(axis=0)
    np.testing.assert_allclose(force, np.array([1.2, -0.8])*3*2*0.8, atol=1e-13)
    doubled = solver.assemble(replace(request, thickness_m=request.thickness_m*2))
    np.testing.assert_allclose(doubled.force, 2*solver.assemble(request).force, atol=1e-13)
    np.testing.assert_allclose(doubled.stiffness.toarray(), 2*base.stiffness.toarray(), atol=1e-9)


def test_nodal_force_control_thickness_changes_displacement(elastic_request):
    # Reuse the consistent traction-equivalent nodal forces, then freeze them as direct N inputs.
    solver = Tri3MechanicsSolver()
    force = solver.assemble(elastic_request).force.reshape((-1, 2))
    ids = np.flatnonzero(np.linalg.norm(force, axis=1))
    request = replace(elastic_request, traction_bcs=(), nodal_force_bcs=(NodalForceBC(ids, force[ids]),))
    first = solver.solve(request)
    second = solver.solve(replace(request, thickness_m=2*request.thickness_m))
    np.testing.assert_allclose(second.state.displacement.values, first.state.displacement.values/2, rtol=1e-12, atol=1e-16)


def test_distinct_material_regions_used(elastic_request):
    mesh = elastic_request.mesh
    centers = np.array([mesh.nodes[c].mean(axis=0) for c in mesh.connectivity])
    regions = (centers[:, 0] > 1.5).astype(int)
    mesh = replace(mesh, material_region_ids=regions)
    assignments = {0: material(70000, 0.3)[0], 1: material(140000, 0.3)[0]}
    request = Tri3SolverRequest.from_mm(mesh, assignments, elastic_request.loading_state, thickness_mm=0.8,
        displacement_bcs=elastic_request.displacement_bcs, traction_bcs=elastic_request.traction_bcs)
    solver = Tri3MechanicsSolver()
    result = solver.solve(request)
    baseline = solver.solve(elastic_request)
    for e, region in enumerate(regions):
        np.testing.assert_allclose(result.assembly.element_stiffness[e], (1+region)*baseline.assembly.element_stiffness[e], rtol=1e-12, atol=1e-9)
    assert np.linalg.norm(result.state.displacement.values-baseline.state.displacement.values) > 1e-8


def test_ill_conditioned_material_contrast_is_reported(elastic_request):
    mesh = elastic_request.mesh
    centers = np.array([mesh.nodes[c].mean(axis=0) for c in mesh.connectivity])
    regions = (centers[:, 0] > 1.5).astype(int)
    mesh = replace(mesh, material_region_ids=regions)
    request = Tri3SolverRequest.from_mm(mesh, {0: material(70000, 0.3)[0], 1: material(7e-14, 0.3)[0]},
        elastic_request.loading_state, thickness_mm=0.8, displacement_bcs=elastic_request.displacement_bcs,
        traction_bcs=elastic_request.traction_bcs)
    with pytest.raises(MechanicsConvergenceError, match="singular|ill-conditioned"):
        Tri3MechanicsSolver().solve(request)


def test_invalid_body_function_and_units():
    with pytest.raises(ContractError):
        BodyForceDensity([1, 2], "N/mm^2")
    with pytest.raises(ContractError):
        BodyForceDensity(lambda q: np.full((len(q), 2), np.nan)).evaluate([[0, 0]])


def test_multiple_material_points_drive_stiffness():
    # A local import through the test module is avoided: the layout is a genuine G=3, E=1 contract.
    from fatigue_dic_rl.geometry.triangles import Tri3Geometry, tri3_shape_functions
    mesh = MeshData([[0, 0], [2, 0], [0, 1]], [[0, 1, 2]], ["tri3"], [0], "mm", "specimen", mesh_id="multi-g")
    points = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    layout = QuadratureLayout(mesh, "tri3-three-point", points, np.full(3, 1/6), tri3_shape_functions(points) @ mesh.nodes,
        np.tile([[2, 0], [0, 1]], (3, 1, 1)), np.full(3, 2), np.zeros(3, dtype=int), (np.arange(3),), np.arange(3))
    damage = layout.field([0.1, 0.2, 0.4], "1")
    request = Tri3SolverRequest.from_mm(mesh, material(70000, 0.3), LoadingState("multi", "static"), thickness_mm=0.8,
        quadrature_layout=layout, damage=damage, displacement_bcs=(DisplacementBC([0, 0, 1], [0, 1, 1], [0, 0, 0], "mm"),),
        nodal_force_bcs=(NodalForceBC([2], [[1, 0]]),))
    solver = Tri3MechanicsSolver()
    result = solver.solve(request)
    undamaged = solver.solve(replace(request, damage=damage.with_values(np.zeros(3))))
    np.testing.assert_allclose(result.assembly.stiffness.toarray(), (1-np.mean(damage.values))*undamaged.assembly.stiffness.toarray(), rtol=1e-12, atol=1e-9)
    assert result.state.stress.values.shape == (3, 3)
    assert result.element_stress.values.shape == (1, 3)
    np.testing.assert_array_equal(result.state.damage.values, [0.1, 0.2, 0.4])


def test_deterministic_sparse_response(elastic_request):
    solver = Tri3MechanicsSolver()
    first, second = solver.solve(elastic_request), solver.solve(elastic_request)
    np.testing.assert_array_equal(first.state.displacement.values, second.state.displacement.values)
    np.testing.assert_array_equal(first.reaction_forces.values, second.reaction_forces.values)


def test_disconnected_components_require_independent_constraints():
    mesh = MeshData([[0, 0], [1, 0], [0, 1], [3, 0], [4, 0], [3, 1]],
                    [[0, 1, 2], [3, 4, 5]], ["tri3", "tri3"], [0, 0], "mm", "specimen")
    first = DisplacementBC([0, 0, 1], [0, 1, 1], [0, 0, 0], "mm")
    second = DisplacementBC([3, 3, 4], [0, 1, 1], [0, 0, 0], "mm")
    request = Tri3SolverRequest.from_mm(mesh, material(70000, 0.3), LoadingState("disconnected", "static"),
        thickness_mm=0.8, displacement_bcs=(first,))
    with pytest.raises(MechanicsConvergenceError, match="component 1"):
        Tri3MechanicsSolver().solve(request)
    result = Tri3MechanicsSolver().solve(replace(request, displacement_bcs=(first, second)))
    np.testing.assert_array_equal(result.state.displacement.values, np.zeros((6, 2)))


def test_hinged_mesh_and_unused_node_rejected():
    for nodes, cells, message in [([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                                  [[0, 1, 2], [0, 3, 4]], "hinge"),
                                 ([[0, 0], [1, 0], [0, 1], [4, 4]], [[0, 1, 2]], "unused")]:
        mesh = MeshData(nodes, cells, ["tri3"]*len(cells), [0]*len(cells), "mm", "specimen")
        request = Tri3SolverRequest.from_mm(mesh, material(70000, 0.3), LoadingState("bad-topology", "static"),
            thickness_mm=0.8, displacement_bcs=(DisplacementBC([0, 0, 1], [0, 1, 1], [0, 0, 0], "mm"),))
        with pytest.raises(ContractError, match=message):
            Tri3MechanicsSolver().solve(request)


def test_explicit_tensor_shear_tag_is_rejected(elastic_request):
    model = elastic_request.materials[0].constitutive_model
    damage = elastic_request.damage
    strain = replace(damage.with_values(np.zeros((damage.count, 3))), metadata={"shear_convention": "tensor"})
    with pytest.raises(ContractError, match="engineering gamma"):
        model.evaluate(strain, damage, elastic_request.materials[0].parameters)


def test_layout_frame_mismatch_rejected(elastic_request):
    layout = elastic_request.quadrature_layout
    bad = replace(layout, mesh=replace(layout.mesh, coordinate_system="camera"))
    with pytest.raises(ContractError, match="layout and request mesh"):
        replace(elastic_request, quadrature_layout=bad)
