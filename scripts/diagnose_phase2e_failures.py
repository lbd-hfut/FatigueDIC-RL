"""Used Phase2D evidence: mechanism diagnosis only, never Phase2E qualification."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition
from fatigue_dic_rl.correction.physics_predictor import CorrectionPredictor
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter
from fatigue_dic_rl.correction.robust_estimation import clean
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference,conditional_noise_monte_carlo
from fatigue_dic_rl.inverse.parameterizations import Parameter,ParameterSpace,UniformDamage
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian


def diagnose(task):
    row,outroot,oldroot=task;out=Path(outroot)/row['case_id'];out.mkdir(parents=True,exist_ok=False);manifest=json.loads((Path(oldroot)/'dataset/manifest.json').read_text());folder=Path(next(x['path'] for x in manifest['records'] if x['case_id']==row['case_id']));pred=[];projections=[];fields={}
    for method in ('B2','B5'):
        acq=load_correction_acquisition(folder,mode='hybrid' if method=='B5' else 'prior',unknown=('d0',) if method=='B5' else ('d0','A','m'));oldspace=acq.predictors[0].parameter_space;E=acq.metadata['material']['E']
        extras=(Parameter('E0',.5*E,1.5*E,'N/mm^2'),Parameter('load_factor',.85,1.15,'1'),Parameter('tx',-.002,.002,'mm'),Parameter('ty',-.002,.002,'mm'),Parameter('theta',-.002,.002,'rad'))
        space=ParameterSpace(oldspace.parameters+extras,dict(oldspace.fixed));ps=tuple(CorrectionPredictor(UniformDamage(),space,p.pattern,law_spec=p.law_spec,load_history=p.load_history,cycle_step=p.cycle_step) for p in acq.predictors);adapter=MultiTrajectoryAdapter(ps,acq.batches)
        physical=dict(row['inference']['candidates'][method]['history']['parameters'],E0=E,load_factor=1.,tx=0.,ty=0.,theta=0.);z=space.encode(physical);y=adapter.whitened_prediction(z);s=centered_jacobian(adapter.whitened_prediction,z);indices=[space.names.index(n) for n in ('E0','load_factor','tx','ty','theta')];pred.append(y);fields[method+'_prediction_white']=y;fields[method+'_J']=s.jacobian;projections.append((s,indices,space))
    difference=pred[0]-pred[1];result=[]
    for method,(s,idx,space) in zip(('B2','B5'),projections):
        for state_nuisance in (False,True):
            columns=idx+[space.names.index('d0')] if state_nuisance else idx;d=project_whitened_difference(difference,s.jacobian[:,columns],numerical_resolution=s.absolute_error_norm);name=method+('_with_state' if state_nuisance else '_observation_nuisance');fields[name+'_perpendicular']=d.projected_difference;fields[name+'_nuisance_basis']=d.nuisance_basis
            result.append(dict(reference=method,state_nuisance=state_nuisance,raw_norm=np.sqrt(d.raw_squared_distance),projected_norm=np.sqrt(d.projected_squared_distance),absorbed_norm=np.sqrt(d.absorbed_squared_distance),rank=d.nuisance_rank,rank_threshold=d.rank_threshold,finite_difference_checked=s.converged,measurement_only_MC=conditional_noise_monte_carlo(d,seed=177000+row['seed']+int(state_nuisance))))
    np.savez_compressed(out/'fields.npz',**fields)
    saved=dict(case_id=row['case_id'],noise_sigma_mm=row['sigma_mm'],selected=row['inference']['selection']['selected'],eligibility=row['inference']['selection']['eligibility'],scores=row['inference']['selection']['candidate_scores'],forecast_ratio_B7_B2=row['methods']['B7']['forecast_damage_rmse'][-1]/row['methods']['B2']['forecast_damage_rmse'][-1],diagnostic_only=True,projections=result)
    (out/'summary.json').write_text(json.dumps(clean(saved),indent=2)+'\n');return clean(saved)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase2d',default='results/phase2d/20261009T014825Z');ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=json.loads((Path(a.phase2d)/'evaluation/records.json').read_text());cases=[r for r in rows if r['split']=='test' and r['family']=='S1' and r['inference']['selection']['eligibility']['B5']];assert [r['case_id'] for r in cases]==['p2d-0042','p2d-0043','p2d-0044','p2d-0045','p2d-0046','p2d-0047','p2d-0062','p2d-0065'];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:result=list(pool.map(diagnose,[(r,str(out),a.phase2d) for r in cases]))
    (out/'summary.json').write_text(json.dumps(dict(cases=result,wall_seconds=time.perf_counter()-t,diagnostic_only=True,qualification_cases=0),indent=2)+'\n');print('eight used-world diagnoses complete')
