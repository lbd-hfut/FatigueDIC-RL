"""Oracle-only separation after formal inference; never feeds model selection."""
import argparse,json
from pathlib import Path
import numpy as np
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition
from fatigue_dic_rl.correction.law_family_benchmarks import ReferenceLawSpecification
from fatigue_dic_rl.correction.hybrid_damage_law import LawSpecification
from fatigue_dic_rl.evaluation.correction_scientific import rollout_from_diagnostic_state,incorrect_state_negative_control
from fatigue_dic_rl.evaluation.correction_metrics import damage_metrics
from fatigue_dic_rl.correction.robust_estimation import clean
from fatigue_dic_rl.training.correction_dataset import generate_correction_case

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--evaluation',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cases=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];rows=[]
    for case in [next(r for r in cases if r['split']=='test' and r['family']==f and r['sigma_mm']==s) for f,s in [('matched',2e-7),('S1',2e-7),('matched',7.5e-4),('S1',7.5e-4)]]:
        folder=Path(case['path']);evaluation=Path(a.evaluation)/'raw'/case['case_id'];formal=json.loads((evaluation/'record.json').read_text());truth=dict(np.load(folder/'synthetic_trajectory.npz',allow_pickle=False));params=json.loads((folder/'synthetic_truth.json').read_text());acq=load_correction_acquisition(folder);cycles=[30.,40.,70.,120.,220.];indices=[int(np.flatnonzero(truth['cycles']==n)[0]) for n in cycles];ref=ReferenceLawSpecification(params['family'],params['A'],params['m'],params['strength']);preds={}
        for model in ['B0','B2','B5','B7']:
            if model not in formal['methods']:continue
            estimate=dict(np.load(evaluation/(model+'_fields.npz'),allow_pickle=False));physical=formal['methods'][model]['parameters'];spec=LawSpecification(mode='hybrid' if 'c0' in physical else 'prior')
            for state_name,current in [('oracle_state',truth['damage'][:,2]),('estimated_state',estimate['current_damage'])]:
                preds[state_name+'_'+model]=np.asarray([rollout_from_diagnostic_state(p,d,physical,spec,cycles) for p,d in zip(acq.predictors,current)])
            preds['estimated_state_correct_law_'+model]=np.asarray([rollout_from_diagnostic_state(p,d,physical,ref,cycles) for p,d in zip(acq.predictors,estimate['current_damage'])])
        preds['oracle_state_correct_law']=np.asarray([rollout_from_diagnostic_state(p,d,params,ref,cycles) for p,d in zip(acq.predictors,truth['damage'][:,2])])
        errors={name:[damage_metrics(v[:,j],truth['damage'][:,indices[j]],truth['physical_weights_mm2'])['rmse'] for j in range(5)] for name,v in preds.items()};assert max(errors['oracle_state_correct_law'])<1e-12
        np.savez_compressed(out/(case['case_id']+'_fields.npz'),**preds,truth=truth['damage'][:,indices]);rows.append(dict(case_id=case['case_id'],family=case['family'],sigma_mm=case['sigma_mm'],oracle_only=True,forecast_damage_rmse=errors))
    # Keep the Phase2C fixed-biased-state diagnostic distinct from deployable estimation.
    special=out/'biased_initial_state_case';generate_correction_case(special,seed=159000,split='scientific',scenario='matched',sigma_mm=5e-5)
    negative=incorrect_state_negative_control(special,out/'biased_initial_state')
    (out/'summary.json').write_text(json.dumps(clean(dict(rows=rows,fixed_biased_state=negative,oracle_only=True)),indent=2)+'\n');print('state/law separation complete',flush=True)
