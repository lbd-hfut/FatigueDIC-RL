from dataclasses import replace

import numpy as np
import pytest

from fatigue_dic_rl.core import FieldLocation
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import BoundaryTag, MeshData


def test_valid_connectivity_and_boundary(mesh):
    assert mesh.n_nodes == 4 and mesh.n_elements == 2
    assert mesh.connectivity[1].tolist() == [0, 2, 3]
    assert mesh.boundary_tags["bottom"].facets.tolist() == [[0, 1]]
    assert mesh.field_reference(FieldLocation.ELEMENT).count == 2


@pytest.mark.parametrize("cells", [[[0, 1, 8], [0, 2, 3]], [[0, -1, 2], [0, 2, 3]],
                                  [[0, 1, 1], [0, 2, 3]], [[0, 1], [0, 2, 3]],
                                  [[0, 1, 2], [2, 1, 0]], [[0.0, 1.0, 2.0], [0, 2, 3]]])
def test_invalid_connectivity(mesh, cells):
    with pytest.raises(ContractError):
        replace(mesh, connectivity=cells)


@pytest.mark.parametrize("updates", [{"nodes": [[0, 0], [1, 0], [2, 0], [0, 1]]},
                                     {"nodes": np.zeros((4, 3))}, {"nodes": [[0, 0], [1, 0], [0, np.nan]]},
                                     {"material_region_ids": [0]}, {"material_region_ids": [-1, 0]},
                                     {"element_types": ["tri3"]}, {"coordinate_units": "px"}])
def test_invalid_mesh_shapes_and_geometry(mesh, updates):
    with pytest.raises(ContractError):
        replace(mesh, **updates)


def test_invalid_boundary_facets(mesh):
    with pytest.raises(ContractError):
        replace(mesh, boundary_tags={"interior": BoundaryTag([], [[0, 2]])})
    with pytest.raises(ContractError):
        replace(mesh, boundary_tags={"wrong": BoundaryTag([50])})


def test_quadrilateral_and_mixed_topology():
    mesh = MeshData([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0]],
                    [[0, 1, 2, 3], [1, 4, 2]], ["quad4", "tri3"], [0, 1], "m", "specimen")
    assert [len(c) for c in mesh.connectivity] == [4, 3]
    with pytest.raises(ContractError):
        replace(mesh, connectivity=[[0, 2, 1, 3], [1, 4, 2]])


def test_clockwise_and_small_scale_geometry(mesh):
    reversed_mesh = replace(mesh, connectivity=[c[::-1] for c in mesh.connectivity])
    assert reversed_mesh.n_elements == 2
    assert replace(mesh, nodes=mesh.nodes*1e-100).n_elements == 2


def test_mesh_owns_immutable_arrays(mesh):
    with pytest.raises(ValueError):
        mesh.nodes[0, 0] = 1
    with pytest.raises(ValueError):
        mesh.nodes.setflags(write=True)
    with pytest.raises(ValueError):
        mesh.connectivity[0][0] = 1
    with pytest.raises(TypeError):
        mesh.boundary_tags["new"] = BoundaryTag([0])


def test_quadrature_reference_is_explicit(mesh):
    with pytest.raises(ContractError):
        mesh.field_reference(FieldLocation.QUADRATURE)
