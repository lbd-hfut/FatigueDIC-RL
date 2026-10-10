"""All predeclared OOD cases; no adaptation/calibration using their truth."""
import argparse,json,time
from pathlib import Path
import tomllib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.synthetic_dataset import generate_case,source_hash,load_labels
from fatigue_dic_rl.evaluation.neural_belief_verification import checkpoint_index,initialize_worker,test_case
from fatigue_dic_rl.evaluation.neural_belief_metrics import summarize_records
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase2b/uniform.toml');ap.add_argument('--initial',required=True);ap.add_argument('--sequential',required=True);ap.add_argument('--calibration',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();c=tomllib.loads(Path(a.config).read_text());c['_generation_source_sha256']=source_hash();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();n=c['evaluation']['ood_cases_per_scenario'];first=c['evaluation']['ood_first_seed']
    scenarios=['q16','q64','q256','q37','regular','clustered','independent','spatial','registration','unknown_loading','cross_mesh','heldout_material','cross_geometry','model_discrepancy','no_noise','missing','irregular_time']
    (out/'frozen_schedule.json').write_text(json.dumps({'scenarios':scenarios,'seeds':list(range(first,first+len(scenarios)*n)),'case_count_per_scenario':n,'calibration_source':a.calibration},indent=2)+'\n')
    tasks=[(c,'ood',first+i*n+j,str(out/'dataset'/s/f'case_{j:03d}'),s) for i,s in enumerate(scenarios) for j in range(n)]
    with ProcessPoolExecutor(max_workers=c['dataset']['workers']) as pool:cases=list(pool.map(generate_case,tasks))
    # Paired same-case/same-noise reference; both members remain wholly OOD.
    cross=[r for r in cases if r['scenario']=='cross_mesh'];pair_jobs=[(c,'ood',r['seed'],str(out/'dataset/cross_mesh_reference'/f'case_{i:03d}'),'cross_mesh_reference') for i,r in enumerate(cross)]
    with ProcessPoolExecutor(max_workers=c['dataset']['workers']) as pool:references=list(pool.map(generate_case,pair_jobs))
    pair_audit=[]
    for fine,reference in zip(cross,references):
        f=load_labels(fine['path']);r=load_labels(reference['path']);np.testing.assert_array_equal(f['physical_parameters'],r['physical_parameters']);np.testing.assert_array_equal(f['coordinates_mm'],r['coordinates_mm']);gap=float(np.max(np.abs(f['exact_displacement_mm']-r['exact_displacement_mm'])));assert gap<1e-12
        pair_audit.append({'seed':fine['seed'],'same_parameters_coordinates_noise':True,'truth_displacement_max_difference_mm':gap})
    cases+=references;scenarios+=['cross_mesh_reference']
    checkpoints=checkpoint_index(a.initial,a.sequential);keys=[k for k in checkpoints if k.startswith(('graph_gru_seed','physics_shape_seed','physics_shape_uncertainty_seed'))];cal=json.loads(Path(a.calibration).read_text())['coefficients'];jobs=[(case,keys,cal,(case['seed']-first)%n<3,str(out/'raw')) for case in cases]
    with ProcessPoolExecutor(max_workers=c['dataset']['workers'],initializer=initialize_worker,initargs=({k:checkpoints[k] for k in keys},)) as pool:rows=[r for result in pool.map(test_case,jobs) for r in result]
    summaries={s:{k:summarize_records([r for r in rows if r['scenario']==s and r['method']==k]) for k in sorted({r['method'] for r in rows if r['scenario']==s})} for s in scenarios}
    paired={}
    for key in keys:
        fine={r['case_seed']:r for r in rows if r['scenario']=='cross_mesh' and r['method']==key and r['status']=='SUCCESS'};reference={r['case_seed']:r for r in rows if r['scenario']=='cross_mesh_reference' and r['method']==key and r['status']=='SUCCESS'};gaps=[fine[s]['forecast_damage_rmse'][1]-reference[s]['forecast_damage_rmse'][1] for s in fine.keys()&reference.keys()];paired[key]={'paired_count':len(gaps),'D40_RMSE_increase_mean':float(np.mean(gaps)) if gaps else None,'per_case_RMSE_difference':gaps}
    (out/'dataset_manifest.json').write_text(json.dumps(cases,indent=2)+'\n');(out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps({'scenarios':summaries,'paired_cross_mesh':paired,'paired_truth_audit':pair_audit,'wall_seconds':time.perf_counter()-t,'source_sha256':source_hash(),'qualification':'OOD diagnostics only; frozen ID calibration coefficients transferred without retuning','material_information':'heldout E value is PUBLIC; unknown load_factor and registration bias are unmodelled discrepancies for all two-parameter estimators','independence':'17 scenarios x12 distinct physics/noise seeds plus12 paired mesh references, NOT additional independent cases; field points not independent trials'},indent=2)+'\n');print('OOD completed',len(rows),'rows',time.perf_counter()-t)
