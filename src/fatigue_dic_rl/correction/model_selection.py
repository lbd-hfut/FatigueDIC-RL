"""Truth-blind model selection using a reserved historical validation frame.

No paths, family labels, oracle states, or future outcomes enter the selector.
The score is a conservative engineering rule, not a Bayes factor or calibrated
likelihood-ratio test: prediction errors of separately fitted models correlate.
"""
from dataclasses import dataclass
import math
import numpy as np
from fatigue_dic_rl.core.validation import ContractError


@dataclass(frozen=True)
class HistoricalCandidate:
    name: str
    validation_square: float
    validation_components: int
    practical_rank: int
    dimension: int
    converged: bool
    boundary: bool
    derivative_checked: bool
    validation_cycle: float = 20.
    fitted_through_cycle: float = 10.
    def __post_init__(self):
        if self.name not in ('B0','B1','B2','B5') or not np.isfinite(self.validation_square) or self.validation_square < 0 or self.validation_components < 1 or not 0 <= self.practical_rank <= self.dimension or not 0 <= self.fitted_through_cycle < self.validation_cycle <= 20.:
            raise ContractError('selector accepts only causal historical candidate diagnostics')
    @property
    def eligible(self):
        return self.converged and self.derivative_checked and not self.boundary and self.practical_rank == self.dimension
    @property
    def score(self):
        return self.validation_square+2*self.practical_rank


@dataclass(frozen=True)
class SelectionRule:
    minimum_gain: float = 9.21
    fluctuation_multiplier: float = 2.
    def __post_init__(self):
        if not np.isfinite([self.minimum_gain,self.fluctuation_multiplier]).all() or min(self.minimum_gain,self.fluctuation_multiplier) <= 0:
            raise ContractError('selection thresholds must be positive')
    def margin(self, components): return max(self.minimum_gain,self.fluctuation_multiplier*math.sqrt(2*components))


def select_model(candidates, rule=None):
    rule = rule or SelectionRule(); rows={c.name:c for c in candidates}
    if len(rows) != len(candidates) or 'B0' not in rows or len({c.validation_components for c in candidates}) != 1:
        raise ContractError('unique candidates with the same held-history likelihood required')
    base=rows['B0']; margin=rule.margin(base.validation_components)
    if not base.converged:
        return dict(selected='B0',status='BASELINE_TENTATIVE',margin=margin,reason='physics fit did not converge')
    param=base
    for name in ('B1','B2'):
        c=rows.get(name)
        if c is not None and c.eligible and param.score-c.score > margin: param=c
    selected=param; corrected=rows.get('B5')
    if corrected is not None and corrected.eligible and param.score-corrected.score > margin: selected=corrected
    return dict(selected=selected.name,status='HISTORY_VALIDATED' if selected.name!='B0' else 'KEEP_PHYSICS_OR_INSUFFICIENT_INFORMATION',
                margin=margin,reason='reserved N20 score, practical data rank and bounds; no future outcomes',
                candidate_scores={n:c.score for n,c in rows.items()},eligibility={n:c.eligible for n,c in rows.items()})


def choose_shrinkage(selection, gate_validation_squares, *, coefficient_norm_square, components, gates=(0.,.25,.5,.75,1.)):
    if selection['selected'] != 'B5':return dict(gate=0.,fallback=selection['selected'],reason='correction not independently history-validated')
    values=np.asarray(gate_validation_squares,float);g=np.asarray(gates,float)
    if values.shape != g.shape or not np.isfinite(values).all() or np.any(values<0) or not np.isfinite(coefficient_norm_square) or coefficient_norm_square<0 or not len(g) or np.any((g<0)|(g>1)):
        raise ContractError('invalid historical gate grid')
    scores=values+2*coefficient_norm_square*g*g
    best=int(np.argmin(scores));margin=selection['margin']
    if scores[0]-scores[best] <= margin:best=0
    return dict(gate=float(g[best]),fallback='B0' if best==0 else None,scores=scores.tolist(),gates=g.tolist(),reason='frozen grid with coefficient shrinkage on reserved N20 only')
