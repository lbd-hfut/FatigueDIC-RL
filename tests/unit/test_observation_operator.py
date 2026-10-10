from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import FieldReference
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import MeshData
from fatigue_dic_rl.observations import ObservationOperator, UnsupportedElementError


def prepare(operator, points):
    return operator.prepare(points, coordinate_units="mm", coordinate_system="specimen")


def test_irregular_affine_interpolation(mesh, nodal_displacement):
    points = np.array([[0.23, 0.12], [1.4, 0.26], [0.51, 0.78], [0.85, 0.6], [0, 0], [-0.08, 0.7]])
    result = prepare(ObservationOperator(mesh), points).apply(nodal_displacement)
    expected = np.column_stack((0.3+0.012*points[:, 0]-0.021*points[:, 1],
                                -0.4+0.008*points[:, 0]+0.015*points[:, 1]))
    assert result.valid_mask.all()
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=4*np.finfo(float).eps)


def test_outside_mask_and_no_extrapolation(mesh, nodal_displacement):
    mapping = prepare(ObservationOperator(mesh), [[3, 0.5], [-0.5, -0.3], [0.3, 0.2]])
    result = mapping.apply(nodal_displacement)
    assert result.valid_mask.tolist() == [False, False, True]
    assert np.isnan(result.values[:2]).all()
    assert np.all(mapping.element_ids[:2] == -1)
    assert np.all(mapping.weights[:2] == 0)


def test_nodes_edges_shape_functions_and_tie(mesh, nodal_displacement):
    operator = ObservationOperator(mesh)
    mapping = prepare(operator, mesh.nodes)
    np.testing.assert_allclose(mapping.apply(nodal_displacement).values, nodal_displacement.values, rtol=0, atol=2e-16)
    np.testing.assert_allclose(mapping.weights.sum(axis=1), 1, rtol=0, atol=2e-16)
    midpoint = (mesh.nodes[0]+mesh.nodes[2])/2
    assert prepare(operator, midpoint[None]).element_ids.tolist() == [0]


def test_cache_determinism_and_immutability(mesh, nodal_displacement):
    operator = ObservationOperator(mesh, cache_size=1)
    points = np.array([[0.3, 0.2]])
    first = prepare(operator, points)
    second = prepare(operator, points.copy())
    assert first is second
    points[:] = 50
    assert first.coordinates.tolist() == [[0.3, 0.2]]
    with pytest.raises(ValueError):
        first.weights.setflags(write=True)
    operator.clear_cache()
    third = prepare(operator, [[0.3, 0.2]])
    assert third is not first
    np.testing.assert_array_equal(first.apply(nodal_displacement).values, third.apply(nodal_displacement).values)


def test_bad_units_frames_shapes_and_same_count_reference(mesh, nodal_displacement):
    operator = ObservationOperator(mesh)
    with pytest.raises(ContractError):
        operator.prepare([[0.3, 0.2]], coordinate_units="m", coordinate_system="specimen")
    with pytest.raises(ContractError):
        operator.prepare([[0.3, 0.2]], coordinate_units="mm", coordinate_system="camera")
    with pytest.raises(ContractError):
        prepare(operator, [[np.nan, 0]])
    with pytest.raises(ContractError):
        prepare(operator, [[0, 0, 0]])
    with pytest.raises(ContractError):
        prepare(operator, [[0.3, 0.2]]).apply(replace(nodal_displacement,
            reference=FieldReference("other", "node", 4, "other")))


def test_invalid_fe_node_propagates_to_mask(mesh, nodal_displacement):
    values = nodal_displacement.values.copy()
    values[1] = np.nan
    field = replace(nodal_displacement, values=values, valid_mask=[True, False, True, True])
    result = prepare(ObservationOperator(mesh), [[1.4, 0.26], [-0.08, 0.7]]).apply(field)
    assert result.valid_mask.tolist() == [False, True]
    assert np.isnan(result.values[0]).all()


def test_quad_mesh_rejected_explicitly():
    mesh = MeshData([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]], ["quad4"], [0], "m", "specimen")
    with pytest.raises(UnsupportedElementError):
        ObservationOperator(mesh)


def test_empty_queries(mesh, nodal_displacement):
    result = prepare(ObservationOperator(mesh), np.empty((0, 2))).apply(nodal_displacement)
    assert result.values.shape == (0, 2)


def test_cache_eviction_and_disabled(mesh):
    operator = ObservationOperator(mesh, cache_size=1)
    first = prepare(operator, [[0.3, 0.2]])
    prepare(operator, [[0.4, 0.2]])
    assert prepare(operator, [[0.3, 0.2]]) is not first
    operator = ObservationOperator(mesh, cache_size=0)
    assert prepare(operator, [[0.3, 0.2]]) is not prepare(operator, [[0.3, 0.2]])
