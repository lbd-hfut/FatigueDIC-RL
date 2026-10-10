"""Tri3 geometry and engineering-shear kinematics, independent of any solver."""

from dataclasses import dataclass

import numpy as np

from fatigue_dic_rl.core.validation import ContractError, array


def tri3_shape_functions(reference_points: np.ndarray) -> np.ndarray:
    points = array(reference_points)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ContractError("reference points must be finite [Q,2]")
    return np.column_stack((1-points.sum(axis=1), points))


@dataclass(frozen=True, eq=False)
class Tri3Geometry:
    jacobian: np.ndarray
    signed_determinant: float
    area: float
    gradients: np.ndarray
    B: np.ndarray

    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> "Tri3Geometry":
        points = array(vertices)
        if points.shape != (3, 2) or not np.isfinite(points).all():
            raise ContractError("Tri3 requires finite vertices [3,2]")
        jacobian = (points[1:]-points[0]).T
        scale = np.max(np.abs(jacobian))
        if not np.isfinite(scale) or scale <= 0:
            raise ContractError("zero or invalid Tri3 scale")
        normalized = jacobian/scale
        normalized_det = float(np.linalg.det(normalized))
        if abs(normalized_det) <= 64*np.finfo(float).eps:
            raise ContractError("zero-area or numerically degenerate Tri3")
        determinant = normalized_det*scale*scale
        if not np.isfinite(determinant) or abs(determinant) == 0:
            raise ContractError("Tri3 area outside the representable numerical range")
        gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]]) @ np.linalg.inv(normalized)/scale
        B = np.zeros((3, 6))
        B[0, 0::2] = gradients[:, 0]
        B[1, 1::2] = gradients[:, 1]
        B[2, 0::2] = gradients[:, 1]
        B[2, 1::2] = gradients[:, 0]
        return cls(array(jacobian), determinant, abs(determinant)/2, array(gradients), array(B))


def tri3_dofs(node_ids: np.ndarray) -> np.ndarray:
    nodes = array(node_ids, kind="int")
    if nodes.shape != (3,) or np.any(nodes < 0) or len(set(nodes.tolist())) != 3:
        raise ContractError("Tri3 DOF map needs three distinct nonnegative node ids")
    return array(np.column_stack((2*nodes, 2*nodes+1)).ravel(), kind="int")


def triangle_gauss_rule(order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Duffy-transformed tensor Gauss rule; weights integrate reference area 1/2.

    Used for body loads and verification errors, separately from material-state
    quadrature. Does not silently change a damage/internal-variable layout.
    """
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 12:
        raise ContractError("triangle integration order must be an integer in [1,12]")
    x, w = np.polynomial.legendre.leggauss(order)
    r, s = np.meshgrid((x+1)/2, (x+1)/2, indexing="ij")
    wr, ws = np.meshgrid(w/2, w/2, indexing="ij")
    return array(np.column_stack((r.ravel(), ((1-r)*s).ravel()))), array((wr*ws*(1-r)).ravel())
