from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import FieldData
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import MeshData, QuadratureLayout
from fatigue_dic_rl.geometry.triangles import Tri3Geometry, tri3_dofs, tri3_shape_functions, triangle_gauss_rule
from fatigue_dic_rl.mechanics.tri3 import tri3_element_stiffness


def test_shape_functions_gradients_and_engineering_shear():
    vertices = np.array([[0, 0], [2, 0], [0, 1]])
    geometry = Tri3Geometry.from_vertices(vertices)
    np.testing.assert_array_equal(tri3_shape_functions([[0, 0], [1, 0], [0, 1]]), np.eye(3))
    np.testing.assert_allclose(geometry.gradients, [[-0.5, -1], [0.5, 0], [0, 1]])
    # u=0.01*y, v=0.02*x: gamma=0.03, tensor epsilon_xy=0.015.
    displacement = np.column_stack((0.01*vertices[:, 1], 0.02*vertices[:, 0])).ravel()
    np.testing.assert_allclose(geometry.B @ displacement, [0, 0, 0.03], atol=1e-16)
    np.testing.assert_array_equal(tri3_dofs([4, 0, 8]), [8, 9, 0, 1, 16, 17])


def test_orientation_area_positive_stiffness_and_rigid_modes():
    nodes = np.array([[0, 0], [2, 0], [0, 1]])
    C = np.diag([10, 10, 4])
    first = Tri3Geometry.from_vertices(nodes)
    reverse = Tri3Geometry.from_vertices(nodes[[0, 2, 1]])
    assert first.signed_determinant == -reverse.signed_determinant
    assert first.area == reverse.area == 1
    K = tri3_element_stiffness(first, C, 0.8)
    reversed_K = tri3_element_stiffness(reverse, C, 0.8)
    order = [0, 1, 4, 5, 2, 3]
    np.testing.assert_allclose(reversed_K[np.ix_(order, order)], K, atol=1e-14)
    eigen = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(eigen) < 1e-12) == 3
    assert np.min(eigen) >= -1e-12
    np.testing.assert_allclose(K, K.T, atol=1e-14)


@pytest.mark.parametrize("vertices", [[[0, 0], [1, 0], [2, 0]], [[0, 0], [0, 0], [0, 0]],
                                     [[0, 0], [1, 0], [1, 1e-16]], [[0, 0, 0]]*3])
def test_degenerate_elements_rejected(vertices):
    with pytest.raises(ContractError):
        Tri3Geometry.from_vertices(vertices)


@pytest.mark.parametrize("C", [np.zeros((3, 3)), np.diag([1, -1, 1]), [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]])
def test_invalid_tangent_rejected(C):
    with pytest.raises(ContractError):
        tri3_element_stiffness(Tri3Geometry.from_vertices([[0, 0], [1, 0], [0, 1]]), C, 1)


def test_gaussian_rule_integrates_polynomial_loads():
    points, weights = triangle_gauss_rule(5)
    np.testing.assert_allclose(weights.sum(), 0.5, atol=1e-15)
    np.testing.assert_allclose(weights @ points[:, 0], 1/6, atol=1e-15)
    np.testing.assert_allclose(weights @ (points[:, 0]*points[:, 1]), 1/24, atol=1e-15)


def test_centroid_layout_and_explicit_location_mapping(mesh, damage):
    layout = QuadratureLayout.tri3_centroid(mesh)
    assert layout.count == mesh.n_elements
    assert layout.reference != mesh.field_reference("element")
    qdamage = layout.project_element_field(damage)
    np.testing.assert_array_equal(qdamage.values, damage.values)
    np.testing.assert_array_equal(layout.element_average(qdamage).values, damage.values)
    np.testing.assert_allclose(layout.physical_weights, [Tri3Geometry.from_vertices(mesh.nodes[c]).area for c in mesh.connectivity])
    with pytest.raises(ContractError):
        damage.assert_same_support(qdamage)
    with pytest.raises(ValueError):
        layout.coordinates.setflags(write=True)
    with pytest.raises(ContractError):
        replace(layout, quadrature_to_element=[1, 0])
    with pytest.raises(ContractError):
        replace(layout, reference_weights=[0.2, 0.5])
    with pytest.raises(ContractError):
        replace(layout, coordinates=layout.coordinates+0.1)


def three_point_layout():
    mesh = MeshData([[0, 0], [2, 0], [0, 1]], [[0, 1, 2]], ["tri3"], [0], "mm", "specimen", mesh_id="three-point-test")
    points = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    return QuadratureLayout(mesh, "tri3-three-point-v1", points, np.full(3, 1/6),
        tri3_shape_functions(points) @ mesh.nodes, np.tile([[2, 0], [0, 1]], (3, 1, 1)), np.full(3, 2.),
        np.zeros(3, dtype=int), (np.arange(3),), np.arange(3))


def test_multiple_points_and_material_state_identity():
    layout = three_point_layout()
    assert layout.mesh.n_elements == 1 and layout.count == 3
    field = layout.field([0.1, 0.2, 0.4], "1")
    np.testing.assert_allclose(layout.element_average(field).values, [0.7/3])
    with pytest.raises(ContractError):
        layout.assert_field(FieldData([0.1, 0.2, 0.4], "node", "1", reference=layout.mesh.field_reference("node")))
