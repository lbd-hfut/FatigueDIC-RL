"""100 predeclared independent seeds per matching/bias case, fixed mechanics."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import tomllib
import numpy as np
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup,generate_truth,observation_series,fatigue_loading
from fatigue_dic_rl.observations.registration import RegistrationError
from fatigue_dic_rl.evaluation.neural_belief_verification import checkpoint_index,initialize_worker,test_case
from fatigue_dic_rl.evaluation.neural_belief_metrics import summarize_records
from fatigue_dic_rl.training.synthetic_dataset import source_hash
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase2b/uniform.toml');ap.add_argument('--initial',required=True);ap.add_argument('--sequential',required=True);ap.add_argument('--calibration',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();c=tomllib.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();first=c['evaluation']['fixed_noise_first_seed'];n=c['evaluation']['fixed_noise_seeds'];seeds=list(range(first,first+n));(out/'frozen_seeds.json').write_text(json.dumps(seeds,indent=2)+'\n')
    model,space,pattern,predictor,x=uniform_setup({'geometry':{'nx':2,'ny':1,'seed':19},'observation':{'count':16}},fatigue=True);actual={'d0':.12,'A':.0015};truths=generate_truth(pattern,model,actual,[0.,10.,20.,30.,40.],x,load_history=fatigue_loading(pattern));cases=[]
    for scenario in ['independent','registration']:
        for seed in seeds:
            case_id=f'fixed-{scenario}-{seed}';p=out/'dataset'/case_id;p.mkdir(parents=True,exist_ok=False);reg=RegistrationError([.0005,-.0003],.0004) if scenario=='registration' else None;frames=observation_series(pattern,truths,x,sigma_mm=5e-5,seed=seed,registration=reg)
            for i,f in enumerate(frames):f.case.measurement.save(p/'observations'/str(i))
            public={'case_id':case_id,'scenario':scenario,'split':'fixed_noise_test','seed':seed,'geometry':{'nx':2,'ny':1,'width_mm':2.,'height_mm':1.,'seed':19},'young_modulus_n_mm2':2000.,'sigma_mm':5e-5,'cycles':[0.,10.,20.,30.,40.],'damage_model':'uniform'}
            (p/'public.json').write_text(json.dumps(public,indent=2)+'\n');np.savez_compressed(p/'synthetic_labels.npz',normalized_parameters=space.encode(actual),physical_parameters=np.array([.12,.0015]),damage=np.array([tt.damage for tt in truths]),exact_displacement_mm=np.array([tt.displacement_dic_mm for tt in truths]),coordinates_mm=x,cycles=np.array([0.,10.,20.,30.,40.]),node_displacement_mm=np.array([tt.response.state.displacement.values for tt in truths]));cases.append({'case_id':case_id,'seed':seed,'scenario':scenario,'path':str(p)})
    cp=checkpoint_index(a.initial,a.sequential);keys=[k for k in cp if k.startswith('physics_shape_uncertainty_')];coeff=json.loads(Path(a.calibration).read_text())['coefficients'];jobs=[(case,keys,coeff,(case['seed']-first)<4,str(out/'raw')) for case in cases]
    with ProcessPoolExecutor(max_workers=c['dataset']['workers'],initializer=initialize_worker,initargs=({k:cp[k] for k in keys},)) as pool:rows=[r for rr in pool.map(test_case,jobs) for r in rr]
    summaries={s:{k:summarize_records([r for r in rows if r['scenario']==s and r['method']==k]) for k in sorted({r['method'] for r in rows})} for s in ['independent','registration']}
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps({'scenarios':summaries,'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'shared_registration_contract':'UNMODELLED bias for both K2 neural and K2 MAP here; Phase2A follow-up uses K5 shared nuisance and is a different-information comparison','calibration':'ID calibration split coefficients transferred, never fitted to these fixed noise/test seeds','seed_contract':'common noise seeds across scenarios for contrast; each scenario has100 independent noise acquisitions; five neural training seeds evaluated separately'},indent=2)+'\n');print('fixed noise completed',time.perf_counter()-t)
