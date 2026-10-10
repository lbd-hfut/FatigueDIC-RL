"""Shared observations/splits for neural and classical estimators.

Only the evaluator opens synthetic labels, AFTER producing parameter estimates.
FEM/uncertainty rollout time is kept separate from network execution time.
"""
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.inference.neural_belief_estimator import NeuralBeliefEstimator
from fatigue_dic_rl.inference.sequential_map import PhysicsBasedBeliefEstimator,SequentialMAPConfig
from fatigue_dic_rl.inference.belief_state import ObservationBatch,ObservationFrame,EstimateRole
from fatigue_dic_rl.inference.physics_predictor import PhysicsForwardAdapter
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian,posterior_uncertainty
from fatigue_dic_rl.inference.forecast import predictions_from_belief
from fatigue_dic_rl.inference.innovation import innovation
from fatigue_dic_rl.inverse.parameterizations import UniformDamage,ParameterSpace
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.training.synthetic_dataset import load_observations,load_labels,load_parameter_labels,public_model,prior_normalized,generate_case,source_hash
from fatigue_dic_rl.training.checkpoint import load_checkpoint
from .belief_metrics import rmse
from .identifiability_verification import clean,points
from .sequential_verification import save_estimator
from .neural_belief_metrics import summarize_records


class CountedTri3(Tri3MechanicsSolver):
    def __init__(self,config=None):super().__init__(config);self.calls=0
    def solve(self,request):self.calls+=1;return super().solve(request)


def counted_predictor(predictor):
    predictor.mechanics=CountedTri3(predictor.mechanics.config);return predictor


def initialize_worker(checkpoints):
    global MODELS
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);MODELS={}
    for key,path in checkpoints.items():MODELS[key]=load_checkpoint(path)


def checkpoint_index(initial_dir,sequential_dir):
    out={}
    for root in [initial_dir,sequential_dir]:
        for p in sorted(Path(root).rglob('checkpoint.pt')):
            out[p.parent.name]=str(p)
    override=Path(sequential_dir)/'checkpoint_selection.json'
    if override.exists():
        selection=json.loads(override.read_text())
        for key,value in selection['overrides'].items():
            if hashlib.sha256(Path(value['path']).read_bytes()).hexdigest()!=value['sha256']:raise RuntimeError('frozen selected checkpoint changed')
            out[key]=value['path']
    return out


def create_neural(predictor,key,calibration=None):
    model,meta=MODELS[key];return NeuralBeliefEstimator(predictor,model,metadata=meta,initial_normalized=prior_normalized(predictor.parameter_space),calibration=calibration)


def calibration_case(task):
    record,keys=task;rows=[]
    for key in keys:
        meta,predictor,frames=load_observations(record['path']);est=create_neural(predictor,key);r=est.assimilate_observation(ObservationBatch(frames))
        # Ground truth is opened only after estimates; no future DIC reads.
        labels=load_parameter_labels(record['path']);means=np.array(r.normalized_after);cov=[]
        # Replay the same observations to record covariance at each prefix.
        _,p2,fs=load_observations(record['path']);e2=create_neural(p2,key)
        for f in fs:
            rr=e2.assimilate_observation(f)
            if rr.filtered_belief is None:break
            cov.append(rr.filtered_belief.uncertainty.covariance_normalized)
        if r.status!='SUCCESS' or len(cov)!=len(frames):rows.append({'key':key,'case_id':record['case_id'],'status':'FAILED','diagnostics':r.diagnostics});continue
        error=means-labels['normalized_parameters'];sd=np.sqrt(np.diagonal(np.array(cov),axis1=-2,axis2=-1))
        rows.append({'key':key,'case_id':record['case_id'],'status':'SUCCESS','cycles':[f.cycle_count for f in frames],'normalized_error':error.tolist(),'normalized_sd':sd.tolist(),'normalized_covariance':np.array(cov).tolist()})
    return rows


def fit_calibration(manifest,checkpoints,output,workers=4):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);keys=[k for k in checkpoints if k.startswith('physics_shape_uncertainty_')];records=[r for r in manifest['records'] if r['split']=='calibration'];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=({k:checkpoints[k] for k in keys},)) as pool:
        raw=[r for result in pool.map(calibration_case,[(r,keys) for r in records]) for r in result]
    coeff={};counts={}
    for key in keys:
        good=[r for r in raw if r['key']==key and r['status']=='SUCCESS'];counts[key]=len(good)
        if len(good)!=len(records):raise RuntimeError('calibration inference failure; all failures retained')
        ratios=np.array([np.abs(r['normalized_error'])/np.array(r['normalized_sd']) for r in good]);n=len(good)
        q=min(1.,np.ceil((n+1)*.95)/n);factor=np.quantile(ratios,q,axis=0,method='higher')/1.959963984540054
        coeff[key]={str(float(c)):a.tolist() for c,a in zip(good[0]['cycles'],factor)}
    result={'method':'per-prefix/per-parameter scale from absolute standardized calibration error quantile; preserve correlations; no test fitting','coefficients':coeff,'counts':counts,'frozen_test_seeds':[r['seed'] for r in manifest['records'] if r['split']=='test'],'checkpoint_sha256':{k:hashlib.sha256(Path(checkpoints[k]).read_bytes()).hexdigest() for k in keys},'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t}
    (out/'records.json').write_text(json.dumps(raw,indent=2)+'\n');(out/'calibration.json').write_text(json.dumps(result,indent=2)+'\n');return result


def evaluation_record(case,key,calibration,*,classical=False,output=None):
    t=time.perf_counter();meta,p,frames=load_observations(case['path']);p=counted_predictor(p);space=p.parameter_space;record={'case_id':case['case_id'],'case_seed':case['seed'],'scenario':case.get('scenario','in_distribution'),'method':key};save=Path(output)/case['case_id']/key if output else None
    if save:save.mkdir(parents=True,exist_ok=False)
    try:
        if classical:
            warm=key=='sequential_warm_map';est=PhysicsBasedBeliefEstimator(p,initializations=[prior_normalized(space)],config=SequentialMAPConfig(warm_start=warm,assimilation_cutoff=20.))
            if key=='physics_prior':pass
            elif key=='independent_frame_inverse' and space.names!=('d0','A'):
                rr=est.assimilate_observation(frames[-1])
                if rr.filtered_belief is None:raise RuntimeError(str(rr.diagnostics))
                if save:save_estimator(save,'classical',est)
            elif key=='independent_frame_inverse':
                model=UniformDamage();adapter=StaticForwardAdapter(model,ParameterSpace(model.parameters),[frames[-1].case],mechanics=p.mechanics);r=invert(InverseObjective(adapter),[np.zeros(1)]).best
                z_current=r.normalized;current=adapter.damage_model.evaluate(p.pattern.quadrature_layout.coordinates,adapter.parameter_space.decode(z_current));u=adapter.predict(z_current)[0][0];record.update(status='SUCCESS',parameter_error=None,parameter_sd=None,current_damage_rmse=rmse(current,load_labels(case['path'])['damage'][2]),current_displacement_rmse_mm=rmse(u[frames[-1].design.active_mask],load_labels(case['path'])['exact_displacement_mm'][2][frames[-1].design.active_mask]),online_seconds=time.perf_counter()-t,fem_solves=p.mechanics.calls,current_role='filtered_current_damage_only',forecast_damage_rmse=None,forecast_displacement_rmse_mm=None,forecast_reason='current inverse has no identified initial damage/fatigue law',estimated_current_damage_mean=float(current.mean()),optimization_cost=r.cost,optimization_nfev=r.nfev)
                record['optimization_converged']=r.converged
                record['optimization_status']='CONVERGED' if r.converged else 'UNCONVERGED_TENTATIVE'
                if save:(save/'record.json').write_text(json.dumps(clean(record),indent=2)+'\n')
                return record
            else:
                for f in frames:
                    r=est.assimilate_observation(f)
                    if r.filtered_belief is None:raise RuntimeError(str(r.diagnostics))
                if save:save_estimator(save,'classical',est)
        else:
            est=create_neural(p,key,calibration.get(key,{}));used=frames[:1] if not est.network.config.recurrent else frames
            r=est.assimilate_observation(ObservationBatch(used))
            record['network_seconds']=r.network_seconds
            if r.status!='SUCCESS':raise RuntimeError(str(r.diagnostics))
            if save:
                ragged={f'innovation_measurement_whitened_{i}':a.measurement_whitened for i,a in enumerate(r.innovations_before_update)}
                np.savez_compressed(save/'update.npz',before=np.array(r.normalized_before),after=np.array(r.normalized_after),preupdate_damage=np.array([a.damage.values for a in r.predictions_before_update]),preupdate_displacement_mm=np.array([a.observations[0].displacement_mm for a in r.predictions_before_update]),filtered_damage=est.belief.current_damage.values,covariance=np.array([]) if est.belief.uncertainty.covariance_normalized is None else est.belief.uncertainty.covariance_normalized,**ragged)
        belief=est.belief;online=time.perf_counter()-t;solves=p.mechanics.calls
        z=belief.normalized_mean;current=p.damage_states(z,[20.])[20.];pred=p.observe(z,frames[-1].design,current,role=EstimateRole.FILTERED if belief.cycle_count==20. else EstimateRole.PREDICTION,conditioned_observation_ids=belief.observation_ids)
        final_mask=frames[-1].design.active_mask
        # Planned sensor coordinates, ROI and acquisition covariance come ONLY
        # from the prefix/public experiment contract. Future dropout is unknown.
        designs=tuple(replace(frames[0].design,observation_id=f"{meta['case_id']}-{i}",cycle_count=float(n),
                      pattern=p.pattern,active_mask=frames[0].case.measurement.roi_mask)
                      for i,n in enumerate(meta['cycles']) if n>20.)
        ft=time.perf_counter();future=est.forecast_future(designs,propagate_uncertainty=False);forecast_time=time.perf_counter()-ft;unc_time=None;future_unc=None
        current_unc=None
        if belief.uncertainty.covariance_normalized is not None:
            ut=time.perf_counter();future_unc=est.forecast_future(designs,propagate_uncertainty=True);unc_time=time.perf_counter()-ut
            current_unc=predictions_from_belief(p,belief,[20.],propagate_uncertainty=True)[0]
        # Forecast is complete before ANY synthetic label or future measurement IO.
        labels=load_labels(case['path']);future_frames=load_observations(case['path'],cutoff=40.)[2][3:]
        future_frames=tuple(ObservationFrame(f.observation_id,replace(f.case,pattern=p.pattern)) for f in future_frames)
        record['forecast_before_label_and_future_measurement_IO']=True
        ds=[rmse(a.damage.values,labels['damage'][i+3]) for i,a in enumerate(future.predictions)];us=[rmse(a.observations[0].displacement_mm[f.design.active_mask],labels['exact_displacement_mm'][i+3][f.design.active_mask]) for i,(a,f) in enumerate(zip(future.predictions,future_frames))]
        covariance=belief.uncertainty.covariance_normalized;parameter_error=None if key=='physics_prior' else belief.parameter_mean-labels['physical_parameters'];sd=None if covariance is None else belief.uncertainty.marginal_standard_deviation_physical
        heldout=points(32,'random',seed=meta['seed']+100000)*[meta['geometry']['width_mm']/2.,meta['geometry']['height_mm']]
        design=replace(frames[-1].design,observation_id='spatial_holdout',coordinates_mm=heldout,active_mask=np.ones(32,dtype=bool),covariance=frames[-1].design.covariance.diagonal(np.full((32,2),meta['sigma_mm'])))
        hold=p.observe(z,design,current).displacement_mm
        if 'node_displacement_mm' in labels:
            mapping=ObservationOperator(p.pattern.mesh).prepare(heldout,coordinate_units='mm',coordinate_system='specimen');truth_hold=np.einsum('qi,qij->qj',mapping.weights,labels['node_displacement_mm'][2][mapping.node_ids])
        else:
            coef=np.linalg.lstsq(np.column_stack([np.ones(len(labels['coordinates_mm'])),labels['coordinates_mm']]),labels['exact_displacement_mm'][2],rcond=None)[0];truth_hold=np.column_stack([np.ones(32),heldout])@coef
        record.update(status='SUCCESS',parameter_estimate=belief.parameter_mean,parameter_error=parameter_error,parameter_sd=sd,covariance_normalized=covariance,current_damage_rmse=rmse(current.damage.values,labels['damage'][2]),current_displacement_rmse_mm=rmse(pred.displacement_mm[final_mask],labels['exact_displacement_mm'][2][final_mask]),fitted_observation_rmse_mm=rmse(pred.displacement_mm[final_mask],frames[-1].case.measurement.observation.displacement[final_mask]),heldout_spatial_rmse_mm=rmse(hold,truth_hold),online_seconds=online,fem_solves=solves,forecast_seconds=forecast_time,uncertainty_seconds=unc_time,forecast_damage_rmse=ds,forecast_displacement_rmse_mm=us,current_role='filtered_after_observation' if belief.cycle_count==20. else 'prediction_from_initial_frame_or_prior',conditioned_ids=belief.observation_ids,conditioned_through_cycle=belief.cycle_count,forecast_origin_cycle=future.origin_cycle_count,uncertainty_status=belief.uncertainty.status,uncertainty_warnings=belief.uncertainty.warnings)
        if sd is not None:
            factor=np.array(calibration.get(key,{}).get(str(float(belief.cycle_count)),np.ones(len(z))))
            record['raw_parameter_sd']=sd/factor
            record['calibration_factor']=factor
        if current_unc is not None and current_unc.damage_standard_deviation is not None:
            record['current_damage_coverage']=float(np.mean(np.abs(current.damage.values-labels['damage'][2])<=1.95996398454*current_unc.damage_standard_deviation))
            record['current_damage_interval_width']=float(np.mean(2*1.95996398454*current_unc.damage_standard_deviation))
        if classical and key!='physics_prior':
            record['optimization_cost']=est.attempts[-1].selected.cost;record['optimization_converged']=est.attempts[-1].status=='CONVERGED';record['optimization_status']=est.attempts[-1].status
        if future_unc is not None and all(a.damage_standard_deviation is not None for a in future_unc.predictions):
            damage_cover=[];mean_cover=[];predictive_cover=[];nis=[];damage_width=[];mean_width=[];predictive_width=[]
            adapter=PhysicsForwardAdapter(p,ObservationBatch(tuple(ObservationFrame(f.observation_id,f.design.empty_case()) for f in future_frames)));s,method=checked_local_jacobian(adapter.whitened_prediction,z);offset=0
            for i,(pr,f) in enumerate(zip(future_unc.predictions,future_frames)):
                mask=f.design.active_mask;truth_y=labels['exact_displacement_mm'][i+3].copy()
                if meta['scenario']=='registration':truth_y+=np.array([.0005,-.0003])+.0004*np.column_stack([-labels['coordinates_mm'][:,1],labels['coordinates_mm'][:,0]])
                err_y=pr.observations[0].displacement_mm[mask]-truth_y[mask];sd_y=pr.displacement_standard_deviation_mm[0][mask];noise=np.sqrt(np.diag(f.case.measurement.covariance.selected_matrix(mask))).reshape(-1,2)
                damage_cover.append(float(np.mean(np.abs(pr.damage.values-labels['damage'][i+3])<=1.95996398454*pr.damage_standard_deviation)))
                damage_width.append(float(np.mean(2*1.95996398454*pr.damage_standard_deviation)))
                mean_cover.append(float(np.mean(np.abs(err_y)<=1.95996398454*sd_y)))
                mean_width.append(float(np.mean(2*1.95996398454*sd_y)))
                observed=f.case.measurement.observation.displacement[mask];predictive_cover.append(float(np.mean(np.abs(pr.observations[0].displacement_mm[mask]-observed)<=1.95996398454*np.sqrt(sd_y**2+noise**2))))
                predictive_width.append(float(np.mean(2*1.95996398454*np.sqrt(sd_y**2+noise**2))))
                size=2*int(mask.sum());evaluation_view=replace(pr.observations[0],active_mask=mask)
                e=innovation(f,evaluation_view,covariance_normalized=covariance,jacobian_whitened=s.jacobian[offset:offset+size],posterior_reliable=s.converged);offset+=size;nis.append(e.normalized_nis)
            record.update(damage_coverage=damage_cover,mean_observation_coverage=mean_cover,predictive_observation_coverage=predictive_cover,normalized_predictive_nis=nis,forecast_damage_interval_width=damage_width,forecast_mean_displacement_interval_width_mm=mean_width,forecast_predictive_displacement_interval_width_mm=predictive_width)
        if save:
            np.savez_compressed(save/'forecast.npz',cycles=np.array([30.,40.]),damage=np.array([a.damage.values for a in future.predictions]),displacement_mm=np.array([a.observations[0].displacement_mm for a in future.predictions]),damage_sd=np.array([a.damage_standard_deviation for a in future_unc.predictions]) if future_unc is not None else np.array([]))
    except Exception as exc:record.update(status='FAILED',error_type=type(exc).__name__,message=str(exc))
    if save:(save/'record.json').write_text(json.dumps(clean(record),indent=2)+'\n')
    return clean(record)


def test_case(task):
    case,keys,calibration,baseline,output=task;rows=[]
    for key in keys:rows.append(evaluation_record(case,key,calibration,output=output))
    if baseline:
        for key in ['physics_prior','independent_frame_inverse','prefix_cold_map','sequential_warm_map']:rows.append(evaluation_record(case,key,{},classical=True,output=output))
    return rows


def evaluate_partition(manifest,checkpoints,calibration,output,*,partition='test',baseline_count=16,workers=4,keys=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);keys=keys or list(checkpoints);cases=[r for r in manifest['records'] if r['split']==partition];t=time.perf_counter()
    tasks=[(r,keys,calibration,i<baseline_count,str(out/'raw')) for i,r in enumerate(cases)]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=({k:checkpoints[k] for k in keys},)) as pool:records=[r for rr in pool.map(test_case,tasks) for r in rr]
    methods=sorted({r['method'] for r in records});summary={k:summarize_records([r for r in records if r['method']==k]) for k in methods}
    comparison_subset={k:summarize_records([r for r in records if r['method']==k and r['case_id'] in {a['case_id'] for a in cases[:baseline_count]}]) for k in methods} if baseline_count else {}
    result={'partition':partition,'case_count':len(cases),'all_seeds':sorted({r['case_seed'] for r in records}),'methods':summary,'fair_comparison_first_frozen_cases':comparison_subset,'baseline_case_count':baseline_count,'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'evidence_contract':'identical case observations/noise/public physics; graph_initial/control consume N0 by definition, all recurrent/MAP use full N0/10/20; forecasts no future assimilation','spatial_holdout_truth':'independent coordinates, affine extension of noiseless uniform-damage FE field; not a general Gaussian-field interpolation'}
    (out/'records.json').write_text(json.dumps(clean(records),indent=2)+'\n');(out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');return result
