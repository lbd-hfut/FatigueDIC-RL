"""Independent synthetic truth evaluation; deliberately outside inverse package."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import json
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array


@dataclass(frozen=True, eq=False)
class SyntheticTruth:
    coordinates_mm: np.ndarray
    damage: np.ndarray
    parameters: dict
    exact_displacement_mm: np.ndarray
    mesh_id: str
    quadrature_reference: str
    mesh_nodes_mm: np.ndarray | None = None
    strain_engineering: np.ndarray | None = None
    stress_n_mm2: np.ndarray | None = None
    reaction_n: np.ndarray | None = None

    def __post_init__(self):
        for name in ("coordinates_mm", "damage", "exact_displacement_mm"):
            object.__setattr__(self, name, array(getattr(self, name), name=name))
        if self.coordinates_mm.shape != (len(self.damage), 2) or not np.isfinite(self.damage).all() or np.any(self.damage < 0) or np.any(self.damage >= 1):
            raise ContractError("invalid synthetic damage truth")
        for name, shape in (("mesh_nodes_mm", self.exact_displacement_mm.shape),
                            ("strain_engineering", (len(self.damage), 3)), ("stress_n_mm2", (len(self.damage), 3)),
                            ("reaction_n", self.exact_displacement_mm.shape)):
            value = getattr(self, name)
            if value is not None:
                value = array(value, name=name)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ContractError("synthetic internal-state support disagrees")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def save(self, folder):
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=False)
        values = {"coordinates_mm": self.coordinates_mm, "damage": self.damage, "exact_displacement_mm": self.exact_displacement_mm}
        values.update({name: getattr(self, name) for name in ("mesh_nodes_mm", "strain_engineering", "stress_n_mm2", "reaction_n")
                       if getattr(self, name) is not None})
        np.savez_compressed(path/"truth.npz", **values)
        (path/"metadata.json").write_text(json.dumps({"schema": "synthetic-truth-1c-v1", "parameters": dict(self.parameters),
            "mesh_id": self.mesh_id, "quadrature_reference": self.quadrature_reference,
            "units": {"coordinates": "mm", "damage": "1", "displacement": "mm", "strain": "1 (engineering shear)",
                      "stress": "N/mm^2", "reaction": "N"}}, indent=2)+"\n")


def residual_metrics(adapter, normalized):
    all_e, whitened, correlations, reaction_residuals = [], [], [], []
    nll = 0.
    for (u, reaction), c, factor in zip(adapter.predict(normalized), adapter.cases, adapter.factors):
        mask = c.measurement.active_mask
        e = (u-c.measurement.observation.displacement)[mask]
        all_e.append(e.ravel())
        whitened.append(factor.whiten(e.ravel()))
        nll += factor.negative_log_likelihood(e.ravel())
        if reaction is not None:
            reaction_residuals.append({"residual_n": reaction-c.reaction.value_n,
                "whitened_residual": (reaction-c.reaction.value_n)/c.reaction.standard_deviation_n})
        x = c.measurement.observation.coordinates[mask]
        if len(x) > 2:
            distance = np.sum((x[:, None]-x[None, :])**2, axis=2)
            np.fill_diagonal(distance, np.inf)
            neighbor = np.argmin(distance, axis=1)
            centered = e-e.mean(axis=0)
            denominator = float(np.sum(centered**2))
            correlations.append(float(np.sum(centered*centered[neighbor])/denominator) if denominator > 1e-28 else None)
    e, w = np.concatenate(all_e), np.concatenate(whitened)
    return {"displacement_rmse_mm": float(np.sqrt(np.mean(e**2))), "weighted_half_square": float(.5*w @ w),
        "whitened_rmse": float(np.sqrt(np.mean(w**2))), "gaussian_negative_log_likelihood": nll,
        "nearest_neighbor_residual_correlation": correlations, "reaction_residuals": reaction_residuals}


def evaluate_recovery(model, space, normalized, truth_parameters, evaluation_coordinates_mm):
    """Truth enters only here, after optimization. Compare fields at same coords."""
    candidate = space.decode(normalized)
    damage = model.evaluate(evaluation_coordinates_mm, candidate)
    true_damage = model.evaluate(evaluation_coordinates_mm, truth_parameters)
    physical_error = {s.name: candidate[s.name]-truth_parameters[s.name] for s in space.parameters}
    return {"parameters": candidate, "parameter_error": physical_error,
            "dimensionless_parameter_rmse": float(np.sqrt(np.mean([(physical_error[s.name]/scale)**2
                          for s, scale in zip(space.parameters, space.scales)]))),
            "damage_field_rmse": float(np.sqrt(np.mean((damage-true_damage)**2)))}
