"""Sparse constant-strain Tri3 backend, with explicit fixed-damage quadrature."""

from dataclasses import dataclass
import logging

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from fatigue_dic_rl.core import FieldData, FieldLocation, MechanicsState, StateRole
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from fatigue_dic_rl.geometry import ElementType, MeshData, QuadratureLayout
from fatigue_dic_rl.geometry.triangles import Tri3Geometry, tri3_dofs, tri3_shape_functions, triangle_gauss_rule
from fatigue_dic_rl.materials import IsotropicPlaneStress, IsotropicPlaneStressMM
from .solver import (MechanicsBackendError, MechanicsConvergenceError, MechanicsSolver,
                     SolverDiagnostics, SolverRequest, SolverResult)
from .tri3_contracts import Tri3SolverConfig, Tri3SolverRequest

logger = logging.getLogger(__name__)


def tri3_element_stiffness(geometry: Tri3Geometry, tangent: np.ndarray, thickness_mm: float) -> np.ndarray:
    C = array(tangent)
    thickness = scalar(thickness_mm, "thickness_mm", minimum=0)
    if C.shape != (3, 3) or not np.isfinite(C).all() or thickness == 0:
        raise ContractError("element stiffness needs a finite [3,3] tangent and positive thickness")
    scale = np.max(np.abs(C))
    if scale == 0 or np.max(np.abs(C-C.T)) > 1e-12*scale:
        raise ContractError("constitutive tangent must be nonzero and symmetric")
    if np.linalg.eigvalsh(C).min() <= 0:
        raise ContractError("constitutive tangent must be positive definite")
    return array(thickness*geometry.area*(geometry.B.T @ C @ geometry.B))


@dataclass(frozen=True, eq=False)
class Tri3Assembly:
    stiffness: csr_matrix
    force: np.ndarray
    element_stiffness: np.ndarray
    element_dofs: np.ndarray
    strain_displacement: np.ndarray
    tangents: np.ndarray
    quadrature_damage: FieldData
    quadrature_layout: QuadratureLayout
    thickness_mm: float


@dataclass(frozen=True)
class Tri3Diagnostics(SolverDiagnostics):
    residual_relative: float = 0
    stiffness_symmetry_relative: float = 0
    minimum_pivot_ratio: float = 1
    free_dof_count: int = 0
    constrained_dof_count: int = 0


@dataclass(frozen=True, eq=False)
class Tri3SolverResult(SolverResult):
    reaction_forces: FieldData
    applied_forces: FieldData
    equilibrium_residual: FieldData
    elastic_strain_energy: float
    element_energy: FieldData
    element_strain: FieldData
    element_stress: FieldData
    assembly: Tri3Assembly
    constrained_dofs: np.ndarray
    energy_units: str = "N*mm"


class Tri3MechanicsSolver(MechanicsSolver):
    """Plane stress only. Mesh/displacement/thickness: mm, C/stress: N/mm^2.

    Supports centroid and explicitly supplied positive multi-point material
    quadrature. Node damage is rejected until explicitly projected by a caller.
    Legacy Pa materials/tractions and thickness_m are converted at named boundaries.
    No damage law or transition is imported or executed.
    """

    def __init__(self, config: Tri3SolverConfig | None = None) -> None:
        self.config = Tri3SolverConfig() if config is None else config
        if not isinstance(self.config, Tri3SolverConfig):
            raise ContractError("config must be Tri3SolverConfig")

    def assemble(self, request: SolverRequest) -> Tri3Assembly:
        if not isinstance(request, SolverRequest):
            raise ContractError("Tri3 backend needs SolverRequest")
        mesh = request.mesh
        if mesh.coordinate_units != "mm":
            raise ContractError("Tri3 backend requires explicitly converted mm mesh coordinates")
        if any(t != ElementType.TRI3 for t in mesh.element_types):
            raise NotImplementedError("Tri3 backend cannot solve quad4 or mixed meshes")
        thickness = request.thickness_m*1000
        if not np.isfinite(thickness) or thickness <= 0:
            raise ContractError("converted thickness_mm must be finite and positive")
        layout = (request.quadrature_layout if isinstance(request, Tri3SolverRequest)
                  else QuadratureLayout.tri3_centroid(mesh))
        if not np.array_equal(layout.material_state_indices, np.arange(layout.count)):
            raise NotImplementedError("this backend uses identity material-state storage indices")
        damage = request.damage
        if damage.location == FieldLocation.ELEMENT:
            qdamage = layout.project_element_field(damage)
        elif damage.location == FieldLocation.QUADRATURE:
            layout.assert_field(damage)
            qdamage = damage
        else:
            raise ContractError("Tri3 damage needs explicit element or quadrature support; node projection is not implicit")
        if np.any(qdamage.values > self.config.damage_max):
            raise ContractError("fixed damage exceeds configured damage_max < 1")
        for value in request.internal_variables.values():
            layout.assert_field(value)
        tangents = np.empty((layout.count, 3, 3))
        for region in sorted(request.materials):
            ids = np.flatnonzero(mesh.material_region_ids[layout.quadrature_to_element] == region)
            if not len(ids):
                continue
            assignment = request.materials[region]
            model = assignment.constitutive_model
            if type(model) not in (IsotropicPlaneStress, IsotropicPlaneStressMM):
                raise NotImplementedError("Phase 1A supports only the two declared isotropic linear plane-stress models")
            # Region subfields have an explicit local support, not a forged full-layout reference.
            from fatigue_dic_rl.core import FieldReference
            reference = FieldReference(f"{layout.reference.space_id}:region-{region}", "quadrature", len(ids), mesh.mesh_id)
            local_damage = FieldData(qdamage.values[ids], "quadrature", "1", reference=reference)
            strain = local_damage.with_values(np.zeros((len(ids), 3)))
            history = {name: FieldData(value.values[ids], "quadrature", value.units, reference=reference)
                       for name, value in request.internal_variables.items()}
            response = model.evaluate(strain, local_damage, assignment.parameters, internal_variables=history)
            unit_factor = {"Pa": 1e-6, "N/mm^2": 1.0}.get(response.tangent_units)
            if unit_factor is None:
                raise ContractError("unsupported tangent units")
            tangents[ids] = response.tangent*unit_factor
        geometry = [Tri3Geometry.from_vertices(mesh.nodes[c]) for c in mesh.connectivity]
        dofs = np.array([tri3_dofs(c) for c in mesh.connectivity])
        matrices = []
        physical_weights = layout.physical_weights
        for e, g in enumerate(geometry):
            ke = np.zeros((6, 6))
            for q in layout.element_to_quadrature[e]:
                # Unit area primitive validates symmetry/positive definiteness; multiply by actual q weight.
                ke += tri3_element_stiffness(g, tangents[q], thickness)*(physical_weights[q]/g.area)
            matrices.append(ke)
        matrices = np.array(matrices)
        rows = np.repeat(dofs, 6, axis=1).ravel()
        cols = np.tile(dofs, (1, 6)).ravel()
        K = coo_matrix((matrices.ravel(), (rows, cols)), shape=(2*mesh.n_nodes, 2*mesh.n_nodes)).tocsr()
        K.sum_duplicates()
        K.sort_indices()
        force = np.zeros((mesh.n_nodes, 2))
        used_nodes = set()
        for bc in request.nodal_force_bcs:
            for node, value in zip(bc.node_ids, bc.values):
                if int(node) in used_nodes:
                    raise ContractError("duplicate nodal force; combine intentional contributions explicitly")
                used_nodes.add(int(node))
                force[node] += value
        used_facets = set()
        for bc in request.traction_bcs:
            factor = 1e-6 if bc.units == "Pa" else 1
            for edge in mesh.boundary_tags[bc.boundary_tag].facets:
                key = tuple(sorted(edge.tolist()))
                if key in used_facets:
                    raise ContractError("duplicate/overlapping traction facets")
                used_facets.add(key)
                length = np.linalg.norm(mesh.nodes[edge[1]]-mesh.nodes[edge[0]])
                force[edge] += factor*bc.values*(thickness*length/2)
        body_force = request.body_force if isinstance(request, Tri3SolverRequest) else None
        if body_force is not None:
            points, weights = triangle_gauss_rule(self.config.load_quadrature_order)
            shape = tri3_shape_functions(points)
            for cell, g in zip(mesh.connectivity, geometry):
                body = body_force.evaluate(shape @ mesh.nodes[cell])
                force[cell] += thickness*abs(g.signed_determinant)*np.einsum("q,qi,qj->ij", weights, shape, body)
        if not np.isfinite(K.data).all() or not np.isfinite(force).all():
            raise MechanicsBackendError("nonfinite stiffness or assembled force")
        return Tri3Assembly(K, array(force.ravel()), array(matrices), array(dofs, kind="int"),
                            array(np.array([g.B for g in geometry])), array(tangents), qdamage, layout, thickness)

    def _constraints(self, request: SolverRequest) -> tuple[np.ndarray, np.ndarray]:
        prescribed = {}
        for bc in request.displacement_bcs:
            for node, component, value in zip(bc.node_ids, bc.components, bc.values):
                dof = int(2*node+component)
                if dof in prescribed:
                    raise ContractError("duplicate prescribed DOF; overlapping node sets must be resolved explicitly")
                prescribed[dof] = float(value)
        ids = np.array(sorted(prescribed), dtype=int)
        return ids, np.array([prescribed[i] for i in ids])

    def _check_rigid_modes(self, mesh: MeshData, constrained: np.ndarray) -> None:
        # Full-edge connectivity: domains meeting at a lone vertex form a hinge,
        # not one rigid body. Reject these unsupported nonconforming assemblies.
        parents = list(range(mesh.n_elements))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        edges = {}
        for e, cell in enumerate(mesh.connectivity):
            for i, j in zip(cell, np.roll(cell, -1)):
                key = tuple(sorted((int(i), int(j))))
                if key in edges:
                    parents[root(e)] = root(edges[key])
                else:
                    edges[key] = e
        components = {}
        for e, cell in enumerate(mesh.connectivity):
            components.setdefault(root(e), set()).update(cell.tolist())
        owned_nodes = set()
        for nodes in components.values():
            if owned_nodes & nodes:
                raise ContractError("components sharing only a vertex form an unsupported hinge")
            owned_nodes.update(nodes)
        if len(owned_nodes) != mesh.n_nodes:
            raise ContractError("mesh contains unused nodes with unsupported free DOFs")
        for number, nodes in enumerate(components.values()):
            nodes = np.array(sorted(nodes), dtype=int)
            coordinates = mesh.nodes[nodes]
            relative = coordinates-coordinates.mean(axis=0)
            relative /= np.max(np.abs(relative))
            modes = np.zeros((2*mesh.n_nodes, 3))
            modes[2*nodes, 0] = 1
            modes[2*nodes+1, 1] = 1
            modes[2*nodes, 2] = -relative[:, 1]
            modes[2*nodes+1, 2] = relative[:, 0]
            rank = np.linalg.matrix_rank(modes[constrained], tol=self.config.rigid_rank_tolerance)
            if rank < 3:
                self._fail(f"underconstrained component {number}: {3-rank} rigid-body mode(s) remain")

    @staticmethod
    def _fail(message: str, residual: float = 0) -> None:
        logger.error("Tri3 solve failed: %s", message)
        raise MechanicsConvergenceError(SolverDiagnostics(False, "scipy-tri3-plane-stress", 0, residual, message))

    def _solve(self, request: SolverRequest) -> Tri3SolverResult:
        assembly = self.assemble(request)
        K, force = assembly.stiffness, assembly.force
        constrained, values = self._constraints(request)
        self._check_rigid_modes(request.mesh, constrained)
        free = np.setdiff1d(np.arange(len(force)), constrained)
        asymmetry = K-K.T
        symmetry = (np.max(np.abs(asymmetry.data)) if asymmetry.nnz else 0)/np.max(np.abs(K.data))
        if symmetry > self.config.symmetry_rtol:
            self._fail("global stiffness is not symmetric")
        u = np.zeros(len(force))
        u[constrained] = values
        pivot_ratio = 1.0
        rhs = force[free]-K[free][:, constrained] @ values
        if len(free):
            reduced = K[free][:, free].tocsc()
            try:
                factor = splu(reduced)
            except RuntimeError as exc:
                self._fail(f"singular reduced stiffness: {exc}")
            pivot_ratio = float(np.min(np.abs(factor.U.diagonal()))/np.max(np.abs(reduced.diagonal())))
            if pivot_ratio <= self.config.minimum_pivot_ratio:
                self._fail(f"numerically singular/ill-conditioned stiffness; pivot ratio={pivot_ratio:.3e}")
            u[free] = factor.solve(rhs)
        if not np.isfinite(u).all():
            raise MechanicsBackendError("sparse solve returned nonfinite displacement")
        residual = K @ u-force
        residual_norm = float(np.max(np.abs(residual[free]))) if len(free) else 0
        internal = K @ u
        force_scale = max(float(np.max(np.abs(internal))), float(np.max(np.abs(force))))
        relative = residual_norm/force_scale if force_scale else 0
        if residual_norm > self.config.residual_atol_n+self.config.residual_rtol*force_scale:
            self._fail("free-DOF equilibrium residual exceeds configured tolerance", residual_norm)
        reactions = np.zeros_like(force)
        reactions[constrained] = residual[constrained]
        element_strain = np.einsum("eij,ej->ei", assembly.strain_displacement, u[assembly.element_dofs])
        layout = assembly.quadrature_layout
        strain = layout.field(element_strain[layout.quadrature_to_element], "1",
                              metadata={"components": "exx,eyy,gamma_xy", "shear_convention": "engineering"})
        stress = layout.field(np.einsum("qij,qj->qi", assembly.tangents, strain.values), "N/mm^2",
                              metadata={"components": "sxx,syy,tau_xy"})
        qenergy = 0.5*assembly.thickness_mm*layout.physical_weights*np.einsum("qi,qi->q", strain.values, stress.values)
        energies = np.array([qenergy[ids].sum() for ids in layout.element_to_quadrature])
        total_energy = float(energies.sum())
        if not np.isfinite(total_energy) or np.any(energies < 0):
            raise MechanicsBackendError("invalid elastic energy")
        mesh = request.mesh

        def node_field(v):
            return FieldData(v.reshape((-1, 2)), "node", "N", reference=mesh.field_reference("node"))

        displacement = FieldData(u.reshape((-1, 2)), "node", "mm", reference=mesh.field_reference("node"))
        state = MechanicsState(mesh.mesh_id, StateRole.PREDICTION, request.damage, request.cycle_count,
                               request.loading_state, displacement, strain, stress, request.internal_variables)
        diagnostics = Tri3Diagnostics(True, "scipy-tri3-plane-stress", 1, residual_norm,
            "equilibrium solve; damage frozen; residual measured on free DOFs in N", relative,
            float(symmetry), pivot_ratio, len(free), len(constrained))
        logger.info("Tri3: V=%d E=%d free=%d residual=%.3e N energy=%.6g N*mm",
                    mesh.n_nodes, mesh.n_elements, len(free), residual_norm, total_energy)
        return Tri3SolverResult(state, diagnostics, node_field(reactions), node_field(force), node_field(residual),
            total_energy, FieldData(energies, "element", "N*mm", reference=mesh.field_reference("element")),
            layout.element_average(strain), layout.element_average(stress), assembly,
            array(constrained, kind="int"))
