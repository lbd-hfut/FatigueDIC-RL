"""Phase2E acquisition factory: N0/10 IO only, public loading thereafter.

No synthetic truth, reference-law family, generation-spec or future observation
imports. Existing parameter boxes, fatigue solver, likelihood and TRF are reused.
"""
from dataclasses import replace
import numpy as np
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition,parameter_space
from fatigue_dic_rl.correction.physics_predictor import CorrectionPredictor
from fatigue_dic_rl.correction.hybrid_damage_law import LawSpecification
from fatigue_dic_rl.correction.robust_estimation import CountedMechanics
from fatigue_dic_rl.inverse.parameterizations import UniformDamage,Parameter,ParameterSpace
from fatigue_dic_rl.inference.belief_state import ObservationBatch,ObservationDesign
from fatigue_dic_rl.observations.covariance import DisplacementCovariance
from fatigue_dic_rl.fatigue import FatigueLoadCycle,LoadingMode
from fatigue_dic_rl.core.validation import ContractError

METHODS=('B0','B1','B2','B5')


def validate_options(options):
    if set(options)-{'shared_registration','load_unknown','material_unknown'} or any(type(v) is not bool for v in options.values()):
        raise ContractError('only public boolean nuisance-condition declarations accepted')


def covariance_at(coordinates,sigma,spatial=False):
    if spatial:return DisplacementCovariance.spatial(coordinates,sigma*sigma*np.array([[1.,.25],[.25,1.6]]),.22)
    return DisplacementCovariance.diagonal(np.tile([sigma,1.25*sigma],(len(coordinates),1)))


def space_for(method,options):
    validate_options(options)
    base=parameter_space(mode='hybrid' if method=='B5' else 'prior',unknown={'B0':('d0',),'B1':('d0','A'),'B2':('d0','A','m'),'B5':('d0',)}[method],shared_registration=options.get('shared_registration',False),load_unknown=options.get('load_unknown',False))
    return ParameterSpace(base.parameters+((Parameter('E0',1500.,2600.,'N/mm^2'),) if options.get('material_unknown') else ()),dict(base.fixed))


def public_prefix(folder,options=None):
    options=options or {};acq=load_correction_acquisition(folder,included_cycles=(0.,10.),cutoff=10.);contract='independent_given_shared_registration' if options.get('shared_registration') else 'conditionally_independent';return acq.metadata,tuple(p.pattern for p in acq.predictors),tuple(replace(b,noise_contract=contract) for b in acq.batches)


def loading(pattern,amplitude=1.,R=-1.,*,future_factor=.4,future_R=-1.):
    common=dict(boundary_condition_refs=('left','right'),loading_mode=LoadingMode.LOAD,load_id=pattern.loading_state.load_id)
    return (FatigueLoadCycle(0.,10.,1.,-1.,-1.,**common),FatigueLoadCycle(10.,10.,amplitude,amplitude*R,R,**common),FatigueLoadCycle(20.,200.,future_factor,future_factor*future_R,future_R,**common))


def predictors_for(patterns,method,meta,options=None,*,amplitude=1.,R=-1.,future_R=-1.,step=2.):
    options=options or {};space=space_for(method,options)
    return tuple(CorrectionPredictor(UniformDamage(),space,p,law_spec=LawSpecification(mode='hybrid' if method=='B5' else 'prior'),load_history=loading(p,amplitude,R,future_factor=meta['future_load_factor'],future_R=future_R),cycle_step=step,mechanics=CountedMechanics()) for p in patterns)


def diagnostic_pattern(pattern,amplitude):
    return replace(pattern,traction_bcs=tuple(replace(b,values=b.values*amplitude) for b in pattern.traction_bcs))


def public_designs(patterns,coordinates,meta,options=None,*,cycle=20.,amplitude=1.,reaction=False,known_healthy=False):
    options=options or {};cov=covariance_at(coordinates,meta['sigma_mm'],meta['noise_model']=='spatial')
    return tuple(ObservationDesign(f"design-{i}-{cycle:g}",diagnostic_pattern(p,amplitude),cycle,coordinates,np.ones(len(coordinates),bool),cov,known_healthy=known_healthy,reaction_node_ids=p.mesh.boundary_tags['left'].node_ids if reaction else None,reaction_standard_deviation_n=.01 if reaction else None) for i,p in enumerate(patterns))


def blank_batches(designs,options=None):
    from fatigue_dic_rl.inference.belief_state import ObservationFrame
    options=options or {};contract='independent_given_shared_registration' if options.get('shared_registration') else 'conditionally_independent'
    return tuple(ObservationBatch((ObservationFrame(d.observation_id,d.empty_case()),),noise_contract=contract) for d in designs)


def combined_batches(prefix,new,options=None):
    options=options or {};contract='independent_given_shared_registration' if options.get('shared_registration') else 'conditionally_independent'
    return tuple(ObservationBatch(a.frames+b.frames,noise_contract=contract) for a,b in zip(prefix,new))
