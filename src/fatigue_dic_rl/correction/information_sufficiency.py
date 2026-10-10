"""Development-only, observation-only information study with original physics.

Loading histories are public, fixed before any measurement. No truth, training,
family labels or evaluation-module imports are accepted by this module.
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.acquisition import public_patterns
from fatigue_dic_rl.correction.designed_acquisition import space_for
from fatigue_dic_rl.correction.physics_predictor import CorrectionPredictor
from fatigue_dic_rl.correction.hybrid_damage_law import LawSpecification
from fatigue_dic_rl.correction.robust_estimation import CountedMechanics
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter
from fatigue_dic_rl.inference.belief_state import ObservationBatch, ObservationFrame, ObservationDesign
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase, ReactionObservation
from fatigue_dic_rl.observations.noise import DICMeasurement
from fatigue_dic_rl.fatigue import FatigueLoadCycle, LoadingMode
from fatigue_dic_rl.inverse.parameterizations import ParameterSpace, UniformDamage

PUBLIC_KEYS = {'schema','case_id','world_id','seed','mesh','material','load_scales',
               'cycles','cutoff','step','sigma_mm','noise_model','history','options',
               'healthy','reaction','future_cycles','units','source_sha256','config_sha256'}


def phase_at(history, n):
    """At an output boundary use the load segment just completed."""
    if n == 0:
        return history[0]
    return next(s for s in history if s['start'] < n <= s['end'])


def history_for(pattern, meta):
    result = []
    for s in meta['history']:
        result.append(FatigueLoadCycle(s['start'], s['end']-s['start'], s['amplitude'],
            s['amplitude']*s['R'], s['R'], ('left','right'), LoadingMode.LOAD,
            load_id=pattern.loading_state.load_id))
    if result[0].N_start != 0 or any(a.N_end != b.N_start for a,b in zip(result,result[1:])):
        raise ContractError('history must be contiguous from the candidate initial state')
    return tuple(result)


def read_observations(folder, *, included_cycles=None):
    folder=Path(folder)
    meta=json.loads((folder/'public.json').read_text())
    if set(meta) != PUBLIC_KEYS or meta['schema'] != 'phase2f-development-v1':
        raise ContractError('unexpected public keys or synthetic truth channel')
    allowed=tuple(meta['cycles']) if included_cycles is None else tuple(included_cycles)
    if not allowed or any(n not in meta['cycles'] or n>meta['cutoff'] for n in allowed):
        raise ContractError('future or unavailable observation requested')
    patterns=public_patterns(meta);batches=[]
    for i,p in enumerate(patterns):
        frames=[]
        for n in allowed:
            k=meta['cycles'].index(n)
            m=DICMeasurement.load(folder/'observations'/str(i)/str(k))
            amplitude=phase_at(meta['history'],n)['amplitude']
            pattern=replace(p,traction_bcs=tuple(replace(b,values=b.values*amplitude) for b in p.traction_bcs))
            force=None
            if n==0 and meta['reaction']:
                r=json.loads((folder/'reactions'/f'{i}.json').read_text())
                force=ReactionObservation(p.mesh.boundary_tags['left'].node_ids,0,r['value_n'],r['standard_deviation_n'])
            frames.append(ObservationFrame(f'{meta["case_id"]}-{i}-N{n:g}',StaticLoadCase(pattern,m,force)))
        if 0 in allowed and meta['healthy']:
            m=DICMeasurement.load(folder/'healthy'/str(i))
            frames.insert(0,ObservationFrame(f'{meta["case_id"]}-{i}-healthy',StaticLoadCase(p,m,known_healthy=True)))
        batches.append(ObservationBatch(tuple(frames)))
    return meta,patterns,tuple(batches)


def predictors(meta, patterns, method, *, step=None):
    space=space_for(method,meta['options'])
    return tuple(CorrectionPredictor(UniformDamage(),
        space,p,law_spec=LawSpecification(mode='hybrid' if method=='B5' else 'prior'),
        load_history=history_for(p,meta),cycle_step=meta['step'] if step is None else step,
        mechanics=CountedMechanics()) for p in patterns)


def adapter_for(meta,patterns,batches,method):
    ps=predictors(meta,patterns,method)
    return MultiTrajectoryAdapter(ps,batches,cutoff=meta['cutoff']),ps


def nominal(space, *, d0=.12):
    return space.encode(dict(d0=d0,A=8e-5,m=1.,c0=0.,c1=0.,E0=2050.,load_factor=1.))


class BoundedProfileAdapter:
    """Fix one PHYSICAL parameter; optimize other bounded parameters via old TRF."""
    def __init__(self,base,name,value):
        if name not in base.parameter_space.names:
            raise ContractError('profile parameter must be a declared unknown')
        spec=next(s for s in base.parameter_space.parameters if s.name==name)
        if not np.isfinite(value) or not spec.lower<=value<=spec.upper:
            raise ContractError('profile value outside physical bounds')
        self.base=base
        self.parameter_space=ParameterSpace(tuple(s for s in base.parameter_space.parameters if s.name!=name),
            dict(base.parameter_space.fixed,**{name:float(value)}))
        self.evidence_ids=base.evidence_ids

    def full(self,z):
        return self.base.parameter_space.encode(self.parameter_space.decode(z))

    def residual(self,z):return self.base.residual(self.full(z))
    def whitened_prediction(self,z):return self.base.whitened_prediction(self.full(z))
    def displacement_prediction(self,z):return self.base.displacement_prediction(self.full(z))
    @property
    def evaluation_count(self):return self.base.evaluation_count


def forecast_fields(meta,ps,fit,coordinates):
    """Consistent filtered state and actual open-loop future; no observation IO."""
    damage=[];current=[];u=[]
    for i,p in enumerate(ps):
        states=p.damage_states(fit.normalized,(meta['cutoff'],*meta['future_cycles']))
        current.append(states[meta['cutoff']].damage.values)
        d=[];rows=[]
        for n in meta['future_cycles']:
            amplitude=phase_at(meta['history'],n)['amplitude']
            pattern=replace(p.pattern,traction_bcs=tuple(replace(b,values=b.values*amplitude) for b in p.pattern.traction_bcs))
            from fatigue_dic_rl.correction.designed_acquisition import covariance_at
            from fatigue_dic_rl.inference.belief_state import EstimateRole
            design=ObservationDesign(f'{meta["case_id"]}-forecast-{i}-{n:g}',pattern,n,coordinates,
                np.ones(len(coordinates),bool),covariance_at(coordinates,meta['sigma_mm']))
            pred=p.observe(fit.normalized,design,states[n],role=EstimateRole.FORECAST,
                           conditioned_observation_ids=fit.evidence_ids)
            d.append(states[n].damage.values);rows.append(pred.displacement_mm)
        damage.append(d);u.append(rows)
    return dict(damage=np.asarray(damage),current_damage=np.asarray(current),u=np.asarray(u),cycles=np.asarray(meta['future_cycles']))
