"""Analytic scientific checks for Phase 1A; these functions perform actual FE solves."""

from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
from scipy.sparse import save_npz

from fatigue_dic_rl.core import FieldData, LoadingState
from fatigue_dic_rl.geometry.triangles import Tri3Geometry, tri3_shape_functions, triangle_gauss_rule
from fatigue_dic_rl.materials import IsotropicPlaneStressMM
from fatigue_dic_rl.mechanics import DisplacementBC, MaterialAssignment, NodalForceBC, TractionBC, MechanicsConvergenceError
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver, Tri3SolverResult
from fatigue_dic_rl.mechanics.tri3_contracts import BodyForceDensity, Tri3SolverConfig, Tri3SolverRequest
from fatigue_dic_rl.observations import DICObservation, ObservationOperator
from .tri3_meshes import rectangle_mesh


def material(young: float, nu: float) -> dict[int, MaterialAssignment]:
    return {0: MaterialAssignment(IsotropicPlaneStressMM(), {"young_modulus_n_per_mm2": young, "poisson_ratio": nu})}


def affine_values(points, coefficients):
    return np.column_stack((np.ones(len(points)), points)) @ np.asarray(coefficients).T


def boundary_affine(mesh, coefficients):
    nodes = np.array(sorted({int(i) for tag in mesh.boundary_tags.values() for i in tag.node_ids}), dtype=int)
    values = affine_values(mesh.nodes[nodes], coefficients)
    return (DisplacementBC(np.repeat(nodes, 2), np.tile([0, 1], len(nodes)), values.ravel(), "mm"),)


def tension_request(mesh, young, nu, thickness, sigma, *, damage=None, control="load", extension_mm=0.01):
    left = mesh.boundary_tags["left"].node_ids
    anchor = int(np.argmin(np.linalg.norm(mesh.nodes, axis=1)))
    bcs = [DisplacementBC(left, np.zeros(len(left), dtype=int), np.zeros(len(left)), "mm"),
           DisplacementBC([anchor], [1], [0], "mm")]
    tractions = ()
    if control == "load":
        tractions = (TractionBC("right", [sigma, 0], "N/mm^2"),)
    elif control == "displacement":
        right = mesh.boundary_tags["right"].node_ids
        bcs.append(DisplacementBC(right, np.zeros(len(right), dtype=int), np.full(len(right), extension_mm), "mm"))
    else:
        raise ValueError("unknown control")
    return Tri3SolverRequest.from_mm(mesh, material(young, nu), LoadingState("uniaxial", "static"),
            thickness_mm=thickness, damage=damage, displacement_bcs=tuple(bcs), traction_bcs=tractions)


def equilibrium_metrics(result: Tri3SolverResult):
    u = result.state.displacement.values.ravel()
    K = result.assembly.stiffness
    force = result.applied_forces.values
    reactions = result.reaction_forces.values
    total = force+reactions
    nodes = result.assembly.quadrature_layout.mesh.nodes
    matrix_energy = float(0.5*u @ (K @ u))
    external = float(u @ force.ravel())
    reaction = float(u @ reactions.ravel())
    free = np.setdiff1d(np.arange(len(u)), result.constrained_dofs)
    virtual = np.zeros_like(u)
    virtual[free] = np.linspace(-0.7, 0.8, len(free))
    virtual_residual = float(virtual @ (K @ u-force.ravel()))
    scale = max(1., abs(matrix_energy), abs(result.elastic_strain_energy))
    energy_gap = abs(matrix_energy-result.elastic_strain_energy)
    work_gap = abs(2*result.elastic_strain_energy-external-reaction)
    if energy_gap > 2e-11*scale or work_gap > 2e-11*scale:
        raise AssertionError("elastic energy or reaction-inclusive work identity failed")
    if abs(virtual_residual) > 1e-8:
        raise AssertionError("free virtual-work equilibrium failed")
    return {"matrix_energy_n_mm": matrix_energy, "integrated_energy_n_mm": result.elastic_strain_energy,
            "energy_abs_gap_n_mm": energy_gap, "force_work_n_mm": external, "reaction_work_n_mm": reaction,
            "work_identity_abs_gap_n_mm": work_gap, "virtual_work_residual_n_mm": virtual_residual,
            "force_balance_n": total.sum(axis=0).tolist(),
            "moment_balance_n_mm": float(np.sum(nodes[:, 0]*total[:, 1]-nodes[:, 1]*total[:, 0]))}


def save_fields(folder: Path | None, name: str, result: Tri3SolverResult):
    if folder is None:
        return
    layout = result.assembly.quadrature_layout
    mesh = layout.mesh
    np.savez_compressed(folder/f"{name}_fields.npz", nodes_mm=mesh.nodes, connectivity=np.array(mesh.connectivity),
        material_regions=mesh.material_region_ids, displacement_mm=result.state.displacement.values,
        quadrature_coordinates_mm=layout.coordinates, reference_points=layout.reference_points,
        reference_weights=layout.reference_weights, jacobians=layout.jacobians,
        physical_weights_mm2=layout.physical_weights, quadrature_to_element=layout.quadrature_to_element,
        material_state_indices=layout.material_state_indices, input_damage=result.state.damage.values,
        quadrature_damage=result.assembly.quadrature_damage.values,
        quadrature_strain_engineering=result.state.strain.values, quadrature_stress_n_mm2=result.state.stress.values,
        element_strain_engineering=result.element_strain.values, element_stress_n_mm2=result.element_stress.values,
        reaction_n=result.reaction_forces.values, applied_force_n=result.applied_forces.values,
        equilibrium_residual_n=result.equilibrium_residual.values, constrained_dofs=result.constrained_dofs,
        element_energy_n_mm=result.element_energy.values)
    save_npz(folder/f"{name}_stiffness.npz", result.assembly.stiffness)
    info = {"mesh_id": mesh.mesh_id, "V": mesh.n_nodes, "E": mesh.n_elements, "G": layout.count,
            "quadrature_reference": layout.reference.space_id, "input_damage_location": result.state.damage.location.value,
            "thickness_mm": result.assembly.thickness_mm, "diagnostics": asdict(result.diagnostics),
            "load_id": result.state.loading_state.load_id, "cycle_count": result.state.cycle_count,
            "prescribed_displacement_mm": result.state.displacement.values.ravel()[result.constrained_dofs].tolist(),
            "units": {"coordinates": "mm", "displacement": "mm", "strain": "1 (engineering gamma_xy)",
                      "stress": "N/mm^2", "reaction": "N", "energy": "N*mm"}}
    (folder/f"{name}_metadata.json").write_text(json.dumps(info, indent=2)+'\n')


def run_patch(config: dict, folder: Path | None = None) -> dict:
    young, nu = config["material"]["young_modulus_n_per_mm2"], config["material"]["poisson_ratio"]
    thickness = config["geometry"]["thickness_mm"]
    coefficients = np.asarray(config["verification"]["affine_coefficients"])
    exact_strain = np.array([coefficients[0, 1], coefficients[1, 2], coefficients[0, 2]+coefficients[1, 1]])
    # Independent analytical stress components, rather than a call to the material implementation.
    expected_stress = np.array([young/(1-nu**2)*(exact_strain[0]+nu*exact_strain[1]),
                               young/(1-nu**2)*(nu*exact_strain[0]+exact_strain[1]),
                               young/(2*(1+nu))*exact_strain[2]])
    solver = Tri3MechanicsSolver(Tri3SolverConfig(**config.get("solver", {})))
    rows = []
    for i, resolution in enumerate(config["verification"]["meshes"]):
        geo = config["geometry"]
        mesh = rectangle_mesh(*resolution, width_mm=geo["width_mm"], height_mm=geo["height_mm"],
                              seed=config["verification"]["mesh_seeds"][i], reverse_alternating=bool(i % 2))
        request = Tri3SolverRequest.from_mm(mesh, material(young, nu), LoadingState("patch", "static"),
            thickness_mm=thickness, displacement_bcs=boundary_affine(mesh, coefficients))
        result = solver.solve(request)
        free = np.setdiff1d(np.arange(2*mesh.n_nodes), result.constrained_dofs)
        if not len(free):
            raise AssertionError("patch test must solve unconstrained interior DOFs")
        reduced = result.assembly.stiffness[free][:, free].toarray()
        eigenvalues = np.linalg.eigvalsh(reduced)
        if eigenvalues[0] <= 0:
            raise AssertionError("constrained patch matrix is not SPD")
        condition = float(np.linalg.cond(reduced))
        full_eigenvalues = np.linalg.eigvalsh(result.assembly.stiffness.toarray())
        zero_modes = int(np.sum(np.abs(full_eigenvalues) < 1e-10*np.max(full_eigenvalues)))
        assert zero_modes == 3
        tolerance_u = 64*np.finfo(float).eps*condition*max(1., np.max(np.abs(affine_values(mesh.nodes, coefficients))))
        tolerance_strain = 256*np.finfo(float).eps*condition
        tolerance_stress = tolerance_strain*young
        exact = affine_values(mesh.nodes, coefficients)
        np.testing.assert_allclose(result.state.displacement.values, exact, rtol=0, atol=tolerance_u)
        np.testing.assert_allclose(result.state.strain.values, np.broadcast_to(exact_strain, result.state.strain.values.shape),
                                   rtol=0, atol=tolerance_strain)
        np.testing.assert_allclose(result.state.stress.values, np.broadcast_to(expected_stress, result.state.stress.values.shape),
                                   rtol=0, atol=tolerance_stress)
        metrics = equilibrium_metrics(result)
        balance_tolerance = 1024*np.finfo(float).eps*condition*max(1., np.max(np.abs(result.reaction_forces.values)))
        np.testing.assert_allclose(metrics["force_balance_n"], 0, rtol=0, atol=balance_tolerance)
        assert abs(metrics["moment_balance_n_mm"]) < 4*balance_tolerance
        row = {"mesh": i, "V": mesh.n_nodes, "E": mesh.n_elements, "free_dofs": len(free),
            "condition_number_2": condition, "displacement_tolerance_mm": tolerance_u,
            "max_displacement_error_mm": float(np.max(np.abs(result.state.displacement.values-exact))),
            "max_strain_error": float(np.max(np.abs(result.state.strain.values-exact_strain))),
            "max_stress_error_n_mm2": float(np.max(np.abs(result.state.stress.values-expected_stress))),
            "minimum_reduced_eigenvalue_n_mm": float(eigenvalues[0]), **metrics}
        row["unconstrained_rigid_modes"] = zero_modes
        rows.append(row)
        save_fields(folder, f"patch_{i}", result)
    mesh = rectangle_mesh(*config["verification"]["rigid_mesh"], width_mm=geo["width_mm"], height_mm=geo["height_mm"],
                          seed=config["verification"]["rigid_seed"])
    rigid = config["verification"]["rigid_coefficients"]
    request = Tri3SolverRequest.from_mm(mesh, material(young, nu), LoadingState("rigid", "static"),
        thickness_mm=thickness, displacement_bcs=boundary_affine(mesh, rigid))
    rigid_result = solver.solve(request)
    assert np.max(np.abs(rigid_result.state.strain.values)) < 2e-13
    assert np.max(np.abs(rigid_result.state.stress.values)) < young*2e-13
    assert rigid_result.elastic_strain_energy < 1e-20
    save_fields(folder, "rigid_body", rigid_result)
    # Nonzero imposed displacement and an independent interior nodal force:
    # reaction work is essential for the scaling identity, not external work alone.
    interior = np.setdiff1d(np.arange(mesh.n_nodes), np.unique(np.concatenate([t.node_ids for t in mesh.boundary_tags.values()])))
    mixed = replace(request, nodal_force_bcs=(NodalForceBC([int(interior[0])], [[0.17, -0.08]]),),
                    displacement_bcs=boundary_affine(mesh, coefficients))
    mixed_result = solver.solve(mixed)
    mixed_metrics = equilibrium_metrics(mixed_result)
    assert abs(mixed_metrics["reaction_work_n_mm"]) > 1e-6
    save_fields(folder, "mixed_work", mixed_result)
    try:
        solver.solve(replace(request, displacement_bcs=()))
    except MechanicsConvergenceError as exc:
        expected_failure = {"status": "EXPECTED_FAILURE", "diagnostics": asdict(exc.diagnostics)}
    else:
        raise AssertionError("unconstrained problem was silently accepted")
    return {"status": "PASS", "patch_rows": rows, "rigid_body": {
        "max_strain": float(np.max(np.abs(rigid_result.state.strain.values))),
        "max_stress_n_mm2": float(np.max(np.abs(rigid_result.state.stress.values))),
        "energy_n_mm": rigid_result.elastic_strain_energy}, "mixed_dirichlet_force_work": mixed_metrics,
        "underconstrained_diagnostic": expected_failure}


def run_uniaxial(config: dict, folder: Path | None = None) -> dict:
    geo, mat = config["geometry"], config["material"]
    mesh = rectangle_mesh(geo["nx"], geo["ny"], width_mm=geo["width_mm"], height_mm=geo["height_mm"], seed=geo["seed"])
    young, nu = mat["young_modulus_n_per_mm2"], mat["poisson_ratio"]
    sigma = config["verification"]["traction_n_per_mm2"]
    result = Tri3MechanicsSolver().solve(tension_request(mesh, young, nu, geo["thickness_mm"], sigma))
    expected = np.column_stack((sigma/young*mesh.nodes[:, 0], -nu*sigma/young*mesh.nodes[:, 1]))
    strain = np.array([sigma/young, -nu*sigma/young, 0])
    np.testing.assert_allclose(result.state.displacement.values, expected, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(result.state.strain.values, np.broadcast_to(strain, result.state.strain.values.shape), atol=1e-13, rtol=1e-10)
    np.testing.assert_allclose(result.state.stress.values, np.broadcast_to([sigma, 0, 0], result.state.stress.values.shape), atol=1e-9, rtol=1e-10)
    expected_force = sigma*geo["height_mm"]*geo["thickness_mm"]
    np.testing.assert_allclose(result.reaction_forces.values.sum(axis=0), [-expected_force, 0], rtol=1e-11, atol=1e-9)
    metrics = equilibrium_metrics(result)
    save_fields(folder, "uniaxial", result)
    # Non-nodal, irregular material coordinates; one outside point, no image correlation.
    fraction = np.array([[0.137, 0.239], [0.431, 0.713], [0.817, 0.317], [0.613, 0.883], [1.2, 0.4]])
    points = fraction*[geo["width_mm"], geo["height_mm"]]
    assert np.min(np.linalg.norm(points[:4, None]-mesh.nodes[None], axis=2)) > 1e-8
    prediction = ObservationOperator(mesh).prepare(points, coordinate_units="mm", coordinate_system="specimen").apply(result.state.displacement)
    np.testing.assert_array_equal(prediction.valid_mask, [True, True, True, True, False])
    exact_dic = np.column_stack((sigma/young*points[:, 0], -nu*sigma/young*points[:, 1]))
    np.testing.assert_allclose(prediction.values[:4], exact_dic[:4], atol=1e-13, rtol=1e-11)
    assert np.isnan(prediction.values[-1]).all() and prediction.units == "mm"
    observation = DICObservation(points, prediction.values, prediction.valid_mask, prediction.valid_mask.astype(float),
        0, "static", "uniaxial", "specimen", "mm", "mm")
    if folder is not None:
        np.savez_compressed(folder/"synthetic_displacement_observation.npz", coordinates_mm=observation.coordinates,
                            displacement_mm=observation.displacement, valid_mask=observation.valid_mask, confidence=observation.confidence)
    return {"status": "PASS", "V": mesh.n_nodes, "E": mesh.n_elements,
        "max_displacement_error_mm": float(np.max(np.abs(result.state.displacement.values-expected))),
        "max_strain_error": float(np.max(np.abs(result.state.strain.values-strain))),
        "reaction_total_n": result.reaction_forces.values.sum(axis=0).tolist(),
        "expected_axial_force_n": expected_force, "transverse_contraction_mm": float(result.state.displacement.values[:, 1].min()),
        "fe_to_dic": {"max_error_mm": float(np.max(np.abs(prediction.values[:4]-exact_dic[:4]))),
                      "valid_mask": prediction.valid_mask.tolist(), "units": prediction.units,
                      "image_correlation_performed": False}, **metrics}


def run_damage(config: dict, folder: Path | None = None) -> dict:
    geo, mat = config["geometry"], config["material"]
    mesh = rectangle_mesh(geo["nx"], geo["ny"], width_mm=geo["width_mm"], height_mm=geo["height_mm"], seed=geo["seed"])
    young, nu = mat["young_modulus_n_per_mm2"], mat["poisson_ratio"]
    solver = Tri3MechanicsSolver()
    zero = tension_request(mesh, young, nu, geo["thickness_mm"], config["verification"]["traction_n_per_mm2"])
    rows = []
    controls = {}
    for control in ["load", "displacement"]:
        base = tension_request(mesh, young, nu, geo["thickness_mm"], config["verification"]["traction_n_per_mm2"],
                               control=control, extension_mm=config["verification"]["extension_mm"])
        baseline = solver.solve(base)
        controls[control] = baseline
        for D in config["verification"]["damage_values"]:
            damage = base.quadrature_layout.field(np.full(base.quadrature_layout.count, D), "1")
            original = damage.values.copy()
            result = solver.solve(replace(base, damage=damage))
            np.testing.assert_array_equal(damage.values, original)
            np.testing.assert_allclose(result.assembly.stiffness.toarray(), (1-D)*baseline.assembly.stiffness.toarray(), atol=1e-9, rtol=1e-12)
            if control == "load":
                np.testing.assert_allclose(result.state.displacement.values, baseline.state.displacement.values/(1-D), atol=1e-13, rtol=1e-10)
            else:
                np.testing.assert_allclose(result.reaction_forces.values, (1-D)*baseline.reaction_forces.values, atol=1e-9, rtol=1e-10)
            right = mesh.boundary_tags["right"].node_ids
            rows.append({"control": control, "damage": D, "effective_E_n_mm2": (1-D)*young,
                "right_mean_u_mm": float(result.state.displacement.values[right, 0].mean()),
                "right_reaction_n": float(result.reaction_forces.values[right, 0].sum()),
                "energy_n_mm": result.elastic_strain_energy, "relative_residual": result.diagnostics.residual_relative})
            save_fields(folder, f"damage_{control}_{D:g}", result)
    x = zero.quadrature_layout.coordinates[:, 0]
    heterogeneous = zero.quadrature_layout.field(np.where(x < geo["width_mm"]/2, 0.1, 0.5), "1")
    result = solver.solve(replace(zero, damage=heterogeneous))
    baseline = controls["load"]
    for e, ids in enumerate(zero.quadrature_layout.element_to_quadrature):
        np.testing.assert_allclose(result.assembly.element_stiffness[e], (1-heterogeneous.values[ids[0]])*baseline.assembly.element_stiffness[e], atol=1e-9, rtol=1e-12)
    assert np.linalg.norm(result.state.displacement.values-baseline.state.displacement.values) > 1e-7
    assert np.linalg.eigvalsh(result.assembly.tangents).min() > 0
    np.testing.assert_array_equal(result.state.damage.values, heterogeneous.values)
    save_fields(folder, "heterogeneous_damage", result)
    return {"status": "PASS", "uniform_rows": rows, "heterogeneous": {
        "damage_min": float(heterogeneous.values.min()), "damage_max": float(heterogeneous.values.max()),
        "displacement_difference_l2_mm": float(np.linalg.norm(result.state.displacement.values-baseline.state.displacement.values)),
        "min_tangent_eigenvalue_n_mm2": float(np.linalg.eigvalsh(result.assembly.tangents).min()),
        "input_damage_preserved": True, **equilibrium_metrics(result)}}


def manufactured_fields(points, amplitude_mm, young, nu):
    x, y = points[:, 0], points[:, 1]
    sx, sy, cx, cy = np.sin(np.pi*x), np.sin(np.pi*y), np.cos(np.pi*x), np.cos(np.pi*y)
    u = np.column_stack((amplitude_mm*sx*sy, np.zeros(len(points))))
    strain = np.column_stack((amplitude_mm*np.pi*cx*sy, np.zeros(len(points)), amplitude_mm*np.pi*sx*cy))
    c11, shear = young/(1-nu**2), young/(2*(1+nu))
    body = amplitude_mm*np.pi**2*np.column_stack(((c11+shear)*sx*sy, -(nu*c11+shear)*cx*cy))
    return u, strain, body


def run_convergence(config: dict, folder: Path | None = None) -> dict:
    mat, verify = config["material"], config["verification"]
    young, nu, amplitude = mat["young_modulus_n_per_mm2"], mat["poisson_ratio"], verify["amplitude_mm"]
    solver = Tri3MechanicsSolver(Tri3SolverConfig(**config.get("solver", {})))
    reference, weights = triangle_gauss_rule(7)
    shape = tri3_shape_functions(reference)
    rows = []
    for n in verify["refinements"]:
        mesh = rectangle_mesh(n, n, seed=verify["seed"], jitter=0.18)
        boundary = np.unique(np.concatenate([t.node_ids for t in mesh.boundary_tags.values()]))
        exact_boundary = manufactured_fields(mesh.nodes[boundary], amplitude, young, nu)[0]
        bcs = (DisplacementBC(np.repeat(boundary, 2), np.tile([0, 1], len(boundary)), exact_boundary.ravel(), "mm"),)
        density = BodyForceDensity(lambda q: manufactured_fields(q, amplitude, young, nu)[2])
        request = Tri3SolverRequest.from_mm(mesh, material(young, nu), LoadingState("manufactured", "static"),
            thickness_mm=config["geometry"]["thickness_mm"], displacement_bcs=bcs, body_force=density)
        result = solver.solve(request)
        displacement_error = strain_error = displacement_norm = strain_norm = 0.
        h = 0.
        for e, cell in enumerate(mesh.connectivity):
            vertices = mesh.nodes[cell]
            geometry = Tri3Geometry.from_vertices(vertices)
            q = shape @ vertices
            exact_u, exact_strain, _ = manufactured_fields(q, amplitude, young, nu)
            prediction_u = shape @ result.state.displacement.values[cell]
            strain_diff = result.element_strain.values[e]-exact_strain
            physical = weights*2*geometry.area
            displacement_error += np.dot(physical, np.sum((prediction_u-exact_u)**2, axis=1))
            # Frobenius norm of symmetric strain: exx^2+eyy^2+2*(gamma/2)^2.
            strain_error += np.dot(physical, np.sum(strain_diff**2*[1, 1, 0.5], axis=1))
            displacement_norm += np.dot(physical, np.sum(exact_u**2, axis=1))
            strain_norm += np.dot(physical, np.sum(exact_strain**2*[1, 1, 0.5], axis=1))
            h = max(h, float(np.linalg.norm(np.roll(vertices, -1, axis=0)-vertices, axis=1).max()))
        row = {"subdivisions": n, "V": mesh.n_nodes, "E": mesh.n_elements, "h_max_mm": h,
            "displacement_L2_mm2": float(np.sqrt(displacement_error)), "strain_L2_mm": float(np.sqrt(strain_error)),
            "relative_displacement_L2": float(np.sqrt(displacement_error/displacement_norm)),
            "relative_strain_L2": float(np.sqrt(strain_error/strain_norm)),
            "free_residual_n": result.diagnostics.residual_norm}
        if rows:
            row["displacement_rate"] = float(np.log(rows[-1]["displacement_L2_mm2"]/row["displacement_L2_mm2"])/np.log(rows[-1]["h_max_mm"]/h))
            row["strain_rate"] = float(np.log(rows[-1]["strain_L2_mm"]/row["strain_L2_mm"])/np.log(rows[-1]["h_max_mm"]/h))
            assert row["displacement_L2_mm2"] < rows[-1]["displacement_L2_mm2"]
            assert row["strain_L2_mm"] < rows[-1]["strain_L2_mm"]
        rows.append(row)
        save_fields(folder, f"manufactured_n{n}", result)
    assert rows[-1]["displacement_rate"] > 1.8
    assert rows[-1]["strain_rate"] > 0.9
    return {"status": "PASS", "manufactured_solution": "u=A*sin(pi*x)*sin(pi*y), v=0; [0,1]^2 mm",
            "body_force": "-div(C*engineering_strain(u)); derived analytically in manufactured_fields",
            "error_integration": "7x7 Duffy Gaussian; independent of centroid material quadrature",
            "rows": rows}


RUNNERS = {"patch": run_patch, "uniaxial": run_uniaxial, "damage": run_damage, "convergence": run_convergence}
