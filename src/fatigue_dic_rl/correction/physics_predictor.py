"""Candidate-specific law injection without changing the validated old predictor.

The initial damage decoder ignores the explicitly declared evolution parameters;
those parameters are consumed ONLY by LawSpecification, not by the initial field.
"""
from dataclasses import dataclass, replace, asdict
from hashlib import sha256
import json
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, scalar
from fatigue_dic_rl.inverse.parameterizations import Parameter
from fatigue_dic_rl.inference.physics_predictor import PhysicsPredictor
from fatigue_dic_rl.fatigue import MaterialHistoryState, FatigueEvolutionSolver, FatigueStepRequest
from .hybrid_damage_law import LawSpecification


@dataclass(frozen=True)
class DamageAndEvolutionParameters:
    initial_decoder: object
    evolution_parameters: tuple

    @property
    def damage_max(self):
        return self.initial_decoder.damage_max

    @property
    def parameters(self):
        return self.initial_decoder.parameters+self.evolution_parameters

    def evaluate(self, coordinates, physical):
        return self.initial_decoder.evaluate(coordinates, physical)


class CorrectionPredictor(PhysicsPredictor):
    def __init__(self, damage_model, parameter_space, pattern, *, law_spec=None, **kwargs):
        for key, default in (('reference_energy_n_mm2', .025), ('law_exponent', 1.), ('law_threshold', 0.), ('law_beta_R', 0.)):
            if key in kwargs and kwargs[key] != default:
                raise ContractError('CorrectionPredictor requires its declared LawSpecification; inherited fixed-law knobs are unsupported')
        self.law_spec = law_spec or LawSpecification()
        ownership = {'m': Parameter('m', .5, 2., '1')}
        ownership.update({f'c{j}': Parameter(f'c{j}', -4., 4., '1') for j in range(8)})
        declared = set(parameter_space.names) | set(parameter_space.fixed)
        wrapper = DamageAndEvolutionParameters(damage_model, tuple(v for k, v in ownership.items() if k in declared))
        super().__init__(wrapper, parameter_space, pattern, **kwargs)

    def provenance(self):
        return dict(super().provenance(), correction_law=asdict(self.law_spec),
            fatigue_exponent_source='physical parameter space m; inherited fixed exponent is a reserved default',
            parameter_ownership={'initial_damage':[s.name for s in self.damage_model.initial_decoder.parameters],
                'evolution':['A', 'm']+[f'c{j}' for j in range(self.law_spec.dimension)]})

    def damage_states(self, normalized, cycles):
        z = np.asarray(normalized, dtype=float)
        physical = self.parameter_space.decode(z)
        times = tuple(sorted(set(scalar(c, 'target cycle', minimum=0) for c in cycles)))
        if not times or not self.load_history or times[-1] > self.load_history[-1].N_end:
            raise ContractError('correction rollout needs targets within a public load history')
        key = (z.tobytes(), times)
        if key in self._history_cache:
            self._history_cache.move_to_end(key)
            return self._history_cache[key].copy()
        layout = self.pattern.quadrature_layout
        initial = MaterialHistoryState.initialize(layout, layout.field(self.damage_model.evaluate(layout.coordinates, physical), '1'), damage_max=self.damage_model.damage_max)
        pattern = self._candidate_pattern(self.pattern, physical)
        law = self.law_spec.build(physical, initial.material_ids)
        solver = FatigueEvolutionSolver(pattern, initial, law, mechanics=self.mechanics, config=self.solver_config)
        result = {}
        try:
            for target in times:
                while solver.committed_state.cycle_count < target:
                    n = solver.committed_state.cycle_count
                    segment = next(c for c in self.load_history if c.N_start <= n < c.N_end)
                    index = int(np.floor((n-segment.N_start)/self.cycle_step+1e-12))+1
                    end = min(segment.N_end, segment.N_start+index*self.cycle_step)
                    if end <= n:
                        raise ContractError('canonical step cannot advance representably')
                    if end <= target:
                        solver.advance(FatigueStepRequest(replace(segment, N_start=n, delta_N=end-n)))
                    else:
                        branch = FatigueEvolutionSolver(pattern, solver.committed_state, law, mechanics=self.mechanics, config=self.solver_config)
                        branch.advance(FatigueStepRequest(replace(segment, N_start=n, delta_N=target-n)))
                        result[target] = branch.committed_state
                        break
                if target not in result:
                    result[target] = solver.committed_state
        except Exception as exc:
            self.failure_diagnostics.append({'parameters':z.tolist(), 'type':type(exc).__name__, 'message':str(exc), 'committed_cycle_before_failure':solver.committed_state.cycle_count})
            raise
        self.rollout_count += 1
        for state in result.values():
            self._history_candidate_keys[state] = z.tobytes()
        self._history_cache[key] = result
        if len(self._history_cache) > 64:
            self._history_cache.popitem(last=False)
        return result.copy()
