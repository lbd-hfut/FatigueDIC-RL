"""Small multi-frame initial-state / fatigue-parameter calibration adapter.

Every prediction creates fresh committed history and runs the verified Phase
1B solver. Calibration cases alone define the objective. Future cases must be
given to a separate adapter by evaluation code, after fitting.
"""
from collections import OrderedDict
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array
from fatigue_dic_rl.fatigue import (MaterialHistoryState, FatigueEvolutionSolver, FatigueLoadCycle,
    FatigueStepRequest, LoadingMode, FatigueLawParameters, SyntheticFatigueDamageLaw, FatigueSolverConfig)
from .forward_adapter import StaticForwardAdapter


class FatigueForwardAdapter(StaticForwardAdapter):
    nuisance_names = {"A"}
    nuisance_units = {"A": "cycle^-1"}

    def __init__(self, damage_model, parameter_space, cases, *, cycle_step=2., reference_energy_n_mm2=.025):
        super().__init__(damage_model, parameter_space, cases)
        self.cycle_step = float(cycle_step)
        self.reference_energy_n_mm2 = float(reference_energy_n_mm2)
        times = [c.measurement.observation.cycle_count for c in self.cases]
        if (not np.isfinite([self.cycle_step, self.reference_energy_n_mm2]).all() or
                self.cycle_step <= 0 or self.reference_energy_n_mm2 <= 0 or
                times != sorted(set(times)) or any(c.pattern is not self.cases[0].pattern for c in self.cases)
                or any(c.known_healthy or c.reaction is not None for c in self.cases)):
            raise ContractError("multi-frame adapter requires ordered unique times, one pattern and DIC-only frames")
        if "A" not in parameter_space.names and "A" not in parameter_space.fixed:
            raise ContractError("declare fatigue A [cycle^-1] as known or unknown")
        if (any(p.lower < 0 for p in parameter_space.parameters if p.name == "A") or
                parameter_space.fixed.get("A", 0.) < 0):
            raise ContractError("fatigue A must be nonnegative cycle^-1")

    def predict(self, normalized):
        z = np.asarray(normalized, dtype=float)
        p = self.parameter_space.decode(z)
        key = z.tobytes()
        if key in self._cache:
            return self._cache[key]
        pattern = self.cases[0].pattern
        layout = pattern.quadrature_layout
        initial = MaterialHistoryState.initialize(layout, layout.field(self.damage_model.evaluate(layout.coordinates, p), "1"),
                                                   damage_max=self.damage_model.damage_max)
        law = SyntheticFatigueDamageLaw({int(r): FatigueLawParameters(p["A"], self.reference_energy_n_mm2)
                                        for r in np.unique(layout.mesh.material_region_ids)})
        solver = FatigueEvolutionSolver(pattern, initial, law, config=FatigueSolverConfig(max_delta_damage_per_step=.03))

        def cycle(start, delta):
            return FatigueLoadCycle(start, delta, 1., -1., -1., ("left", "right"), LoadingMode.LOAD,
                                    load_id=pattern.loading_state.load_id)

        result = []
        for c, mapping in zip(self.cases, self.mappings):
            target = c.measurement.observation.cycle_count
            while solver.committed_state.cycle_count < target:
                current = solver.committed_state.cycle_count
                step = solver.advance(FatigueStepRequest(cycle(current, min(self.cycle_step, target-current))))
                maximum = step.substeps[-1].maximum
            if target == 0:
                _, maximum, _ = solver.evaluate_committed(cycle(0, self.cycle_step))
            u = mapping.apply(maximum.state.displacement).values
            result.append((array(u), None))
            self.solver_diagnostics.append({"cycle_count": target, "residual_norm": maximum.diagnostics.residual_norm})
        self.evaluation_count += 1
        self._cache[key] = tuple(result)
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return self._cache[key]
