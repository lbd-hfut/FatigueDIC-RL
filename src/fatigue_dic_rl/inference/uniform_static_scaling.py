"""Optional exact scaling for a narrowly declared static, uniform-damage problem.

This is an observation adapter, not a new mechanics solver. Reference fields are
computed with the unchanged Tri3 solver. No fatigue history is cached or scaled.
Nonzero prescribed displacement, heterogeneous materials, registration, other
damage decoders and constitutive subclasses are explicitly unsupported.
"""
from collections import OrderedDict
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.inverse.parameterizations import UniformDamage
from fatigue_dic_rl.materials import IsotropicPlaneStressMM


class UniformStaticScalingAdapter(StaticForwardAdapter):
    def __init__(self, damage_model, parameter_space, cases, *, mechanics=None):
        if type(damage_model) is not UniformDamage:
            raise ContractError("exact static scaling requires the uniform damage decoder")
        names = set(parameter_space.names) | set(parameter_space.fixed)
        if names - {"d0", "E0", "load_factor"} or not {"d0", "E0"} <= names:
            raise ContractError("exact scaling supports only d0, E0 and load_factor")
        # Validate the entire domain before invoking a solver.
        for p in parameter_space.parameters:
            if p.name in ("E0", "load_factor") and p.lower <= 0:
                raise ContractError("scaling bounds require positive modulus and load")
            if p.name == "d0" and (p.lower < 0 or p.upper >= damage_model.damage_max):
                raise ContractError("scaling bounds must lie below damage maximum")
        for c in cases:
            p = c.pattern
            if p.cycle_count != 0 or p.internal_variables or any(np.any(b.values != 0) for b in p.displacement_bcs):
                raise ContractError("scaling requires N0 and homogeneous zero displacement BCs")
            values = []
            for assignment in p.materials.values():
                if type(assignment.constitutive_model) is not IsotropicPlaneStressMM:
                    raise ContractError("scaling requires the verified mm plane-stress model")
                if set(assignment.parameters) != {"young_modulus_n_per_mm2", "poisson_ratio"}:
                    raise ContractError("scaling requires explicit homogeneous isotropic material")
                values.append(tuple(sorted(assignment.parameters.items())))
            if not values or len(set(values)) != 1:
                raise ContractError("heterogeneous materials are outside the scaling contract")
        super().__init__(damage_model, parameter_space, cases, mechanics=mechanics)
        self.reference_results = tuple(self.mechanics.solve(c.pattern) for c in self.cases)
        self.reference_moduli = tuple(next(iter(c.pattern.materials.values())).parameters[
            "young_modulus_n_per_mm2"] for c in self.cases)
        self._cache = OrderedDict()
        self.cache_hits = 0

    def scaled_fields(self, normalized, case_index):
        """Diagnostic fields with explicit scaling, not a fabricated SolverResult."""
        p = self.parameter_space.decode(normalized)
        d = 0. if self.cases[case_index].known_healthy else p["d0"]
        self.damage_model.evaluate(self.cases[case_index].pattern.quadrature_layout.coordinates, p)
        f, E = p.get("load_factor", 1.), p["E0"]
        if E <= 0 or f <= 0:
            raise ContractError("scaling requires positive modulus and load")
        q = f*self.reference_moduli[case_index]/(E*(1-d))
        r = self.reference_results[case_index]
        return dict(displacement=r.state.displacement.values*q,
                    strain=r.state.strain.values*q, stress=r.state.stress.values*f,
                    reaction=r.reaction_forces.values*f,
                    applied_force=r.applied_forces.values*f,
                    elastic_strain_energy=r.elastic_strain_energy*q*f)

    def predict(self, normalized):
        z = np.asarray(normalized, dtype=float)
        self.parameter_space.decode(z)
        key = z.tobytes()
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        result = []
        for i, (c, mapping) in enumerate(zip(self.cases, self.mappings)):
            fields = self.scaled_fields(z, i)
            u = mapping.apply(self.reference_results[i].state.displacement.with_values(
                fields["displacement"])).values
            reaction = None if c.reaction is None else float(fields["reaction"][
                c.reaction.node_ids, c.reaction.component].sum())
            result.append((array(u), reaction))
        self.evaluation_count += 1
        self._cache[key] = tuple(result)
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return self._cache[key]

    def analytic_whitened_jacobian(self, normalized):
        """Exact derivatives in the existing dimensionless parameter coordinates."""
        p = self.parameter_space.decode(normalized)
        parts = []
        for (u, reaction), c, factor in zip(self.predict(normalized), self.cases, self.factors):
            multiplier = {"d0": 0. if c.known_healthy else 1/(1-p["d0"]),
                          "E0": -1/p["E0"], "load_factor": 1/p.get("load_factor", 1.)}
            columns = [factor.whiten(u[c.measurement.active_mask].ravel())*
                       multiplier[name]*scale for name, scale in zip(
                           self.parameter_space.names, self.parameter_space.scales)]
            parts.append(np.column_stack(columns))
            if reaction is not None:
                parts.append(np.array([[reaction/c.reaction.standard_deviation_n*
                    scale/p.get("load_factor", 1.) if name == "load_factor" else 0.
                    for name, scale in zip(self.parameter_space.names, self.parameter_space.scales)]]))
        return np.vstack(parts)
