import numpy as np
import pytest

from fatigue_dic_rl.core import FieldData, FieldLocation, LoadingState, MechanicsState, StateRole
from fatigue_dic_rl.geometry import BoundaryTag, MeshData


@pytest.fixture
def mesh():
    return MeshData([[0, 0], [2, 0], [1.7, 1.2], [-0.2, 1]],
                    [[0, 1, 2], [0, 2, 3]], ["tri3", "tri3"], [0, 1], "mm", "specimen",
                    boundary_tags={"bottom": BoundaryTag([0, 1], [[0, 1]])}, mesh_id="test-mesh")


@pytest.fixture
def damage(mesh):
    return FieldData([0.0, 0.2], FieldLocation.ELEMENT, "1",
                     reference=mesh.field_reference(FieldLocation.ELEMENT))


@pytest.fixture
def state(mesh, damage):
    return MechanicsState(mesh.mesh_id, StateRole.BELIEF, damage, 0, LoadingState("load-1", "peak"))


@pytest.fixture
def nodal_displacement(mesh):
    coefficients = np.array([[0.3, 0.012, -0.021], [-0.4, 0.008, 0.015]])
    values = np.column_stack((np.ones(mesh.n_nodes), mesh.nodes)) @ coefficients.T
    return FieldData(values, FieldLocation.NODE, "mm", reference=mesh.field_reference(FieldLocation.NODE))
