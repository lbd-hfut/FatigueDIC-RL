"""Truth is opened for scoring only after frozen-prefix fits and all forecasts."""
from pathlib import Path
from dataclasses import asdict,replace
import json,time
import numpy as np
from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.inference.belief_state import MechanicsBelief,EstimateRole
from fatigue_dic_rl.inference.forecast import forecast_future
from fatigue_dic_rl.inference.uncertainty import posterior_uncertainty
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.fatigue import MaterialHistoryState,EnergyAmplitudeDriver
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter,fit_correction
from fatigue_dic_rl.correction.hybrid_damage_law import LawSpecification
from .correction_metrics import damage_metrics,correction_factor_metrics
from .identifiability_verification import clean


class CountedMechanics(Tri3MechanicsSolver):
    def __init__(self):super().__init__();self.calls=0
    def solve(self,request):self.calls+=1;return super().solve(request)


METHODS={
    'B0_physics_prior':{'mode':'prior','unknown':('d0',)},
    'B1_refit_A':{'mode':'prior','unknown':('d0','A')},
    'B2_refit_Am':{'mode':'prior','unknown':('d0','A','m')},
    'B3_single_step_K2':{'mode':'hybrid','dimension':2,'included_cycles':(0.,10.)},
    'B5_multistep_K2':{'mode':'hybrid','dimension':2},
    'B5_multistep_K4':{'mode':'hybrid','dimension':4},
    'B5_multistep_K8':{'mode':'hybrid','dimension':8},
    'ablation_no_physics_K2':{'mode':'direct','dimension':2},
    'ablation_no_history_K2':{'mode':'hybrid','dimension':2,'included_cycles':(20.,)},
    'ablation_spatial_K2':{'mode':'hybrid','dimension':2,'law_spec':LawSpecification(mode='hybrid',dimension=2,basis_kind='spatial')},
}


def make_belief(predictor, fit, acquisition):
    origin=max(f.cycle_count for b in acquisition.batches for f in b.frames)
    state=predictor.damage_states(fit.normalized,[origin])[origin]
    return MechanicsBelief(predictor.parameter_space.names,fit.normalized,
        [fit.physical[n] for n in predictor.parameter_space.names],fit.uncertainty,state.damage,origin,EstimateRole.FILTERED,
        fit.evidence_ids,None,json.dumps({'physics_model_id':predictor.model_id,'role':'filtered','algorithm':'multi-trajectory_prefix_TRF',
        'noise_contract':'independent_given_shared_registration' if 'tx' in fit.physical else 'conditionally_independent',
        'uncertainty_qualification':'NOT_QUALIFIED_under_constitutive_discrepancy'}))


def evaluate_correction_method(folder, method, output, *, checkpoint=None, full_horizon=True, extra_options=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter()
    options=dict(METHODS.get(method,{}));options.update(extra_options or {})
    if method.startswith('B4_supervised'):
        from fatigue_dic_rl.training.correction_supervised import load_residual_checkpoint
        model,meta=load_residual_checkpoint(checkpoint)
        options={'mode':'hybrid','dimension':meta['dimension'],'coefficients':model.coefficients.detach().numpy(),'unknown':('d0',),
            'law_spec':LawSpecification(mode='hybrid',dimension=meta['dimension'],alpha=meta['alpha'],basis_kind=meta['basis_kind'])}
    if method == 'correct_physics_oracle_reference':
        # Declared extra-information reference, not an observation-only method.
        known = json.loads((Path(folder)/'synthetic_truth.json').read_text())
        options={'mode':'structural_reference','unknown':('d0',),
            'known_physics_override':{'A':known['A'],'m':known['m']},
            'law_spec':LawSpecification(mode='structural_reference',beta=known['beta'],spatial_strength=known['spatial_strength'],structural_power=known.get('structural_power',1))}
    acquisition=load_correction_acquisition(folder,**options)
    for p in acquisition.predictors:p.mechanics=CountedMechanics()
    adapter=MultiTrajectoryAdapter(acquisition.predictors,acquisition.batches)
    # All initialization is based on declared public nominal values, never truth.
    nominal={'d0':.12,**acquisition.metadata['nominal'],**{n:0. for n in adapter.parameter_space.names if n.startswith('c')},'tx':0.,'ty':0.,'theta':0.,'load_factor':1.}
    initial=adapter.parameter_space.encode(nominal)
    if options.get('mode')=='direct':
        # The squared direct-feature hazard has zero derivative at a=0.
        initial=adapter.parameter_space.encode(dict(nominal,c0=1.,c1=0.))
    fit=fit_correction(adapter,initializations=[initial],max_nfev=80)
    fit_seconds=fit.wall_seconds;prefix_predictions=[a.predict(fit.normalized) for a in adapter.adapters]
    cycles=[30.,40.,50.,70.,120.,220.] if full_horizon else [30.,40.]
    public_designs=acquisition.future_designs(cycles=cycles)
    forecasts=[];filtered=[];heldpred=[];diagnostics=[];current_u=[];current_stress=[];current_strain=[]
    future_stress=[];future_strain=[];future_hazard=[];future_original=[];future_factors=[];future_physical_u=[]
    # Complete every physics prediction BEFORE any labels or future observation IO.
    for p,b,designs in zip(acquisition.predictors,acquisition.batches,public_designs):
        belief=make_belief(p,fit,acquisition)
        forecasts.append(forecast_future(p,belief,designs,propagate_uncertainty=False))
        state=p.damage_states(fit.normalized,[20.])[20.]; filtered.append(state.damage.values)
        # Public evaluation coordinates are derived from the prefix domain without truth.
        rng=np.random.default_rng(acquisition.metadata['seed']+202)
        bounds=np.asarray(acquisition.metadata['mesh']['nodes']);x=rng.uniform([.03,.03],[1.97,.97],(64,2))*[np.ptp(bounds[:,0])/2.,np.ptp(bounds[:,1])]
        pattern=p._candidate_pattern(p.pattern,fit.physical);response=p.mechanics.solve(replace(pattern,damage=state.damage,cycle_count=20.,loading_state=LoadingState(pattern.loading_state.load_id,'max')))
        mapping=ObservationOperator(pattern.mesh).prepare(x,coordinate_units='mm',coordinate_system='specimen');heldpred.append(mapping.apply(response.state.displacement).values)
        current_mapping=ObservationOperator(pattern.mesh).prepare(b.frames[-1].design.coordinates_mm,coordinate_units='mm',coordinate_system='specimen')
        current_u.append(current_mapping.apply(response.state.displacement).values)
        current_stress.append(response.state.stress.values);current_strain.append(response.state.strain.values)
        cycle=replace(p.load_history[0],N_start=20.,delta_N=2.)
        lo=p.mechanics.solve(cycle.request(pattern,state.damage,20.,'min'));hi=p.mechanics.solve(cycle.request(pattern,state.damage,20.,'max'));driving=EnergyAmplitudeDriver().evaluate(lo,hi)
        law=p.law_spec.build(fit.physical,state.material_ids);rate=law.hazard(driving,cycle,state)
        if hasattr(law,'last_diagnostics') and law.last_diagnostics is not None:
            d=law.last_diagnostics;diagnostics.append({'basis':d['basis'],'coefficients':d['coefficients'],'factor':d['factor'],'sensitivity':d['coefficient_sensitivity_log_hazard'],'original_hazard':d['prior_hazard'].values,'corrected_hazard':rate.values})
        else:diagnostics.append({'basis':np.empty((state.layout.count,0)),'coefficients':np.empty(0),'factor':np.ones(state.layout.count),'sensitivity':np.empty((state.layout.count,0)),'original_hazard':rate.values,'corrected_hazard':rate.values})
        fs=[];fe=[];fh=[];fo=[];ff=[];fu=[]
        for prediction in forecasts[-1].predictions:
            instant=MaterialHistoryState.initialize(p.pattern.quadrature_layout,prediction.damage,damage_max=.95,cycle_count=prediction.cycle_count)
            cyc=replace(p.load_history[1],N_start=prediction.cycle_count,delta_N=2.)
            lo_f=p.mechanics.solve(cyc.request(pattern,instant.damage,instant.cycle_count,'min'))
            hi_f=p.mechanics.solve(cyc.request(pattern,instant.damage,instant.cycle_count,'max'))
            fu.append(current_mapping.apply(hi_f.state.displacement).values)
            dri_f=EnergyAmplitudeDriver().evaluate(lo_f,hi_f);ha=law.hazard(dri_f,cyc,instant)
            fs.append(hi_f.state.stress.values);fe.append(hi_f.state.strain.values);fh.append(ha.values)
            if hasattr(law,'last_diagnostics') and law.last_diagnostics is not None:
                fo.append(law.last_diagnostics['prior_hazard'].values);ff.append(law.last_diagnostics['factor'])
            else:fo.append(ha.values);ff.append(np.ones(instant.layout.count))
        future_stress.append(fs);future_strain.append(fe);future_hazard.append(fh);future_original.append(fo);future_factors.append(ff);future_physical_u.append(fu)
    inference_done=time.perf_counter()-t;fem_calls=sum(p.mechanics.calls for p in acquisition.predictors)
    # Explicit boundary between estimator operations and evaluation-only truth.
    with np.load(Path(folder)/'synthetic_trajectory.npz',allow_pickle=False) as a:truth={k:a[k].copy() for k in a.files}
    truth_parameters=json.loads((Path(folder)/'synthetic_truth.json').read_text())
    indices=[int(np.flatnonzero(truth['cycles']==n)[0]) for n in cycles];current_index=int(np.flatnonzero(truth['cycles']==20.)[0]);weights=truth['physical_weights_mm2']
    from fatigue_dic_rl.observations.registration import RegistrationError
    true_registration=RegistrationError([5e-5,-3e-5],4e-5).field(truth['coordinates_mm']) if acquisition.metadata['shared_registration'] else np.zeros_like(truth['coordinates_mm'])
    current=np.asarray(filtered);forecast_damage=np.asarray([[p.damage.values for p in f.predictions] for f in forecasts]);forecast_u=np.asarray([[p.observations[0].displacement_mm for p in f.predictions] for f in forecasts])
    future_observed=[]
    from fatigue_dic_rl.observations.noise import DICMeasurement
    for i in range(len(acquisition.predictors)):
        future_observed.append([DICMeasurement.load(Path(folder)/'observations'/str(i)/str(int(np.flatnonzero(np.asarray(acquisition.metadata['cycles'])==n)[0]))).observation.displacement for n in cycles])
    record={'method':method,'case_id':acquisition.metadata['case_id'],'seed':acquisition.metadata['seed'],'scenario':truth_parameters['scenario'],
        'status':fit.status,'parameters':fit.physical,'parameter_error':{n:fit.physical[n]-truth_parameters[n] for n in ('d0','A','m') if n in fit.physical},
        'data_rank':fit.uncertainty.data_rank,'parameter_dimension':len(fit.normalized),'weak_directions':fit.uncertainty.weak_directions,
        'posterior_approximation_status':fit.uncertainty.status,'uncertainty_qualification':'NOT_QUALIFIED; Fisher is a local information diagnostic only',
        'current_damage':damage_metrics(current,truth['damage'][:,current_index],weights),
        'current_state_role':'filtered' if forecasts[0].origin_cycle_count==20. else 'prediction_from_earlier_prefix',
        'current_displacement_rmse_mm':float(np.sqrt(np.mean((np.asarray(current_u)-truth['displacement_mm'][:,current_index])**2))),
        'current_stress_rmse_n_mm2':float(np.sqrt(np.mean((np.asarray(current_stress)-truth['stress'][:,current_index])**2))),
        'current_strain_rmse':float(np.sqrt(np.mean((np.asarray(current_strain)-truth['strain'][:,current_index])**2))),
        'current_hazard_rmse_synthetic_only':float(np.sqrt(np.mean((np.asarray([d['corrected_hazard'] for d in diagnostics])-truth['hazard'][:,current_index])**2))),
        'forecast_cycles':cycles,'forecast_cycle_spans':[n-forecasts[0].origin_cycle_count for n in cycles],
        'comparison_cycles_from20':[n-20. for n in cycles],
        'forecast_damage_rmse':[damage_metrics(forecast_damage[:,j],truth['damage'][:,indices[j]],weights)['rmse'] for j in range(len(cycles))],
        'forecast_displacement_rmse_mm':[float(np.sqrt(np.mean((np.asarray(future_physical_u)[:,j]-truth['displacement_mm'][:,indices[j]])**2))) for j in range(len(cycles))],
        'forecast_registered_observation_rmse_mm':[float(np.sqrt(np.mean((forecast_u[:,j]-truth['displacement_mm'][:,indices[j]]-true_registration)**2))) for j in range(len(cycles))],
        'displacement_metric_frames':'physical FEM displacement versus physical truth; registered predicted observation versus noiseless registered observation truth',
        'heldout_observation_rmse_mm':float(np.sqrt(np.mean((forecast_u-np.asarray(future_observed))**2))),
        'spatial_heldout_displacement_rmse_mm':float(np.sqrt(np.mean((np.asarray(heldpred)-truth['heldout_displacement_mm'][:,current_index])**2))),
        'whitened_prefix_rmse':float(np.sqrt(np.mean(adapter.residual(fit.normalized)**2))), 'data_cost':fit.data_cost,'regularization_cost':fit.regularization_cost,
        'fem_solves':fem_calls,'fit_seconds':fit_seconds,'online_seconds':inference_done,'wall_seconds':time.perf_counter()-t,
        'coefficients':[fit.physical[n] for n in fit.physical if n.startswith('c') and n[1:].isdigit()],
        'correction_factor':correction_factor_metrics(np.concatenate([d['factor'] for d in diagnostics])),
        'conditioning_ids':fit.evidence_ids,'conditioning_through_cycle':forecasts[0].origin_cycle_count,
        'forecast_before_truth_IO':method!='correct_physics_oracle_reference',
        'forecast_before_damage_truth_and_future_DIC_IO':True,
        'training_information':'oracle known physics reference; d0 is DIC-estimated' if method=='correct_physics_oracle_reference' else 'privileged synthetic hazard labels' if checkpoint else 'observed DIC prefix only',
        'checkpoint':str(checkpoint) if checkpoint else None}
    record['irreversibility_violations']=int(np.sum(np.diff(np.concatenate([current[:,None],forecast_damage],axis=1),axis=1)<-1e-13))
    record['forecast_damage_max_error']=float(np.max(abs(forecast_damage-truth['damage'][:,indices])))
    record['forecast_hazard_rmse_synthetic_only']=float(np.sqrt(np.mean((np.asarray(future_hazard)-truth['hazard'][:,indices])**2)))
    record['forecast_stress_rmse_n_mm2']=float(np.sqrt(np.mean((np.asarray(future_stress)-truth['stress'][:,indices])**2)))
    record['forecast_strain_rmse']=float(np.sqrt(np.mean((np.asarray(future_strain)-truth['strain'][:,indices])**2)))
    if float(np.std(truth['damage'][:,current_index],axis=1).max())<1e-10:
        record['damage_hotspot_location']='N/A: uniform damage has no unique hotspot'
    else:
        record['damage_hotspot_location']={'predicted_mm':truth['quadrature_coordinates_mm'][np.argmax(current,axis=1)].tolist(),
            'truth_mm':truth['quadrature_coordinates_mm'][np.argmax(truth['damage'][:,current_index],axis=1)].tolist(),
            'meaning':'quadrature maximum only; localization is mesh sensitive'}
    (out/'record.json').write_text(json.dumps(clean(record),indent=2,allow_nan=False)+'\n')
    (out/'optimization.json').write_text(json.dumps(clean(asdict(fit.optimization)),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'information.npz',jacobian=fit.uncertainty.jacobian_whitened,fisher=fit.uncertainty.data_fisher,null_directions=fit.uncertainty.data_null_directions,weak_directions=fit.uncertainty.weak_directions,singular_values=fit.uncertainty.singular_values_data,finite_difference_error=fit.uncertainty.finite_difference_error,posterior_covariance_is_qualified=np.array(False))
    fields={'current_damage':current,'forecast_damage':forecast_damage,'forecast_displacement_mm':forecast_u,'cycles':np.asarray(cycles),'physical_weights_mm2':weights,
        'current_displacement_mm':np.asarray(current_u),'current_stress_n_mm2':np.asarray(current_stress),'current_strain':np.asarray(current_strain),
        'forecast_stress_n_mm2':np.asarray(future_stress),'forecast_strain':np.asarray(future_strain),'forecast_physical_displacement_mm':np.asarray(future_physical_u),
        'forecast_original_hazard':np.asarray(future_original),'forecast_corrected_hazard':np.asarray(future_hazard),'forecast_factors':np.asarray(future_factors)}
    for i,d in enumerate(diagnostics):fields.update({f'{k}_{i}':v for k,v in d.items()})
    np.savez_compressed(out/'fields.npz',**fields)
    return clean(record)
