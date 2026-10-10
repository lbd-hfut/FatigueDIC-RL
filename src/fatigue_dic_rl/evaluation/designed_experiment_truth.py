"""Privileged synthetic measurement producer/scorer, outside inference imports."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from fatigue_dic_rl.correction.designed_acquisition import loading,diagnostic_pattern,covariance_at
from fatigue_dic_rl.correction.law_family_benchmarks import ReferenceLawSpecification
from fatigue_dic_rl.fatigue import MaterialHistoryState,FatigueEvolutionSolver,FatigueStepRequest,FatigueSolverConfig,EnergyAmplitudeDriver
from fatigue_dic_rl.mechanics import MaterialAssignment
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.noise import simulate_measurement
from fatigue_dic_rl.observations.registration import RegistrationError
from fatigue_dic_rl.observations.covariance import DisplacementCovariance
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase,ReactionObservation
from fatigue_dic_rl.inference.belief_state import ObservationFrame,ObservationBatch
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver


def actual_rollout(folder,patterns,design,coordinates,*,cycles=(20.,),future_R=-1.,step=2.):
    meta=json.loads((Path(folder)/'public.json').read_text());params=json.loads((Path(folder)/'synthetic_truth.json').read_text());reference=ReferenceLawSpecification(params['family'],params['A'],params['m'],params['strength']);result={k:[] for k in ('damage','u','reaction','hazard','energy')};cost=[]
    if max(meta['load_scales'])*10*design['amplitude']*params['load_bias']>35.:raise ValueError('synthetic experiment exceeded declared 35 N/mm^2 nominal-stress bound')
    for pattern in patterns:
        actual=replace(pattern,materials={r:MaterialAssignment(a.constitutive_model,dict(a.parameters,young_modulus_n_per_mm2=params['actual_E'])) for r,a in pattern.materials.items()},traction_bcs=tuple(replace(b,values=b.values*params['load_bias']) for b in pattern.traction_bcs));layout=actual.quadrature_layout;state=MaterialHistoryState.initialize(layout,layout.field(np.full(layout.count,params['d0']),'1'),damage_max=.95);solver=FatigueEvolutionSolver(actual,state,reference.build({},state.material_ids),config=FatigueSolverConfig(max_delta_damage_per_step=.03));history=loading(pattern,design['amplitude'],design['R'],future_factor=meta['future_load_factor'],future_R=future_R);mapping=ObservationOperator(pattern.mesh).prepare(coordinates,coordinate_units='mm',coordinate_system='specimen');rows={k:[] for k in result};energy_cost=0.
        for target in cycles:
            while solver.committed_state.cycle_count<target:
                n=solver.committed_state.cycle_count;segment=next(s for s in history if s.N_start<=n<s.N_end);end=min(target,segment.N_end,n+step);advanced=solver.advance(FatigueStepRequest(replace(segment,N_start=n,delta_N=end-n)))
                if solver.committed_state.damage.values.max()>=.6:raise ValueError('synthetic experiment exceeded preregistered D<.6 safety bound')
            state=solver.committed_state;segment=history[0] if target<=10. else history[1] if target<=20. else history[2];cycle=replace(segment,N_start=target,delta_N=step);hi=solver.mechanics.solve(cycle.request(actual,state.damage,target,'max'));lo=solver.mechanics.solve(cycle.request(actual,state.damage,target,'min'));driver=EnergyAmplitudeDriver().evaluate(lo,hi);u=mapping.apply(hi.state.displacement).values
            for key,value in dict(damage=state.damage.values,u=u,reaction=float(hi.reaction_forces.values[actual.mesh.boundary_tags['left'].node_ids,0].sum()),hazard=reference.build({},state.material_ids).hazard(driver,cycle,state).values,energy=driver.energy_amplitude.values).items():rows[key].append(value)
        for key in result:result[key].append(rows[key])
    return dict(**{k:np.asarray(v) for k,v in result.items()},cycles=np.asarray(cycles),weights=patterns[0].quadrature_layout.physical_weights,parameters=params)


def acquisition_at_twenty(folder,patterns,design,pool,indices,options,*,future_R=-1.,reaction=False):
    meta=json.loads((Path(folder)/'public.json').read_text());both=actual_rollout(folder,patterns,design,pool,cycles=(10.,20.),future_R=future_R);truth={k:v[:,1:2] if k in ('damage','u','reaction','hazard','energy') else v for k,v in both.items()};truth['cycles']=np.array([20.]);truth['initial_experiment_damage']=both['damage'][:,0];cov=covariance_at(pool,meta['sigma_mm'],meta['noise_model']=='spatial');reg=RegistrationError([5e-5,-3e-5],4e-5) if meta['shared_registration'] else None;batches=[];idx=np.asarray(indices)
    for i,p in enumerate(patterns):
        # Same full-pool noise realization across counterfactual design arms.
        m=simulate_measurement(pool,truth['u'][i,0],cov,seed=meta['seed']*313+i,registration=reg,cycle_count=20.,load_id=p.loading_state.load_id);o=replace(m.observation,coordinates=m.observation.coordinates[idx],displacement=m.observation.displacement[idx],valid_mask=m.observation.valid_mask[idx],confidence=m.observation.confidence[idx]);channels=np.array([2*j+k for j in idx for k in (0,1)]);m=replace(m,observation=o,covariance=DisplacementCovariance(cov.matrix[np.ix_(channels,channels)]),roi_mask=np.ones(len(idx),bool));force=None
        if reaction:force=ReactionObservation(p.mesh.boundary_tags['left'].node_ids,0,truth['reaction'][i,0]+np.random.default_rng(meta['seed']*317+i).normal(0.,.01),.01)
        case=StaticLoadCase(diagnostic_pattern(p,design['amplitude']),m,force);batches.append(ObservationBatch((ObservationFrame(f"acquired-{i}-N20",case),),noise_contract='independent_given_shared_registration' if options.get('shared_registration') else 'conditionally_independent'))
    return tuple(batches),truth
