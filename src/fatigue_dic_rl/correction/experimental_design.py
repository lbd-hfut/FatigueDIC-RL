"""Non-RL selection of public, budget-matched numerical experiment candidates."""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError


@dataclass(frozen=True)
class DesignScore:
    design_id: str
    fisher_score: float
    projected_separation: float
    measurement_cost: int
    fatigue_cycles_per_specimen: float
    safe: bool
    derivative_checked: bool

    def __post_init__(self):
        if not self.design_id or not np.isfinite([self.fisher_score,self.projected_separation,self.fatigue_cycles_per_specimen]).all() or self.projected_separation<0 or self.measurement_cost<1 or self.fatigue_cycles_per_specimen<0:
            raise ContractError('finite public numerical design diagnostics required')


def choose_experiment(candidates,*,strategy,seed=0):
    if strategy not in ('fisher','discrimination','random') or isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ContractError('declared deterministic design strategy and seed required')
    rows=list(candidates)
    if not rows or len({r.design_id for r in rows})!=len(rows) or len({(r.measurement_cost,r.fatigue_cycles_per_specimen) for r in rows})!=1:
        raise ContractError('unique candidates with equal measurement/cycle budgets required')
    eligible=[r for r in rows if r.safe and r.derivative_checked]
    if not eligible:return dict(status='ABSTAIN',design_id=None,reason='all candidates unsafe or numerical sensitivities unreliable')
    eligible=sorted(eligible,key=lambda r:r.design_id)
    if strategy=='random':chosen=eligible[int(np.random.default_rng(seed).integers(len(eligible)))]
    else:chosen=max(eligible,key=lambda r:r.fisher_score if strategy=='fisher' else r.projected_separation)
    return dict(status='DESIGNED',design_id=chosen.design_id,strategy=strategy,selection_reads_measurements=False,reason='estimated-prefix predictions and public candidate budget only')
