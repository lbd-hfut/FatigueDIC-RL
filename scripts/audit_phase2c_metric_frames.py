"""Re-score frozen predictions in matching frames, without any re-fit or rollout."""
import argparse,json,time
from pathlib import Path
import numpy as np
from fatigue_dic_rl.observations.registration import RegistrationError

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();rows=[]
    cases={r['case_id']:Path(r['path']) for f in ('dataset','ood_dataset') for r in json.loads((root/f/'manifest.json').read_text())['records']}
    for suite in ('test','fair_calibration','ood','nuisance','oracle_reference','spatial_basis_followup','serial_cost'):
        for r in json.loads((root/suite/'records.json').read_text()):
            folder=root/suite/'raw'/r['case_id']/r['method'] if suite!='serial_cost' else root/suite/r['method']
            case=cases[r['case_id']];meta=json.loads((case/'public.json').read_text())
            with np.load(case/'synthetic_trajectory.npz',allow_pickle=False) as a,np.load(folder/'fields.npz',allow_pickle=False) as p:
                ids=[int(np.flatnonzero(a['cycles']==n)[0]) for n in r['forecast_cycles']];truth=a['displacement_mm'][:,ids];x=a['coordinates_mm'];params=r['parameters'];pred=p['forecast_displacement_mm']
                bias=RegistrationError([params.get('tx',0.),params.get('ty',0.)],params.get('theta',0.)).field(x)
                physical=pred-bias
                true_bias=RegistrationError([5e-5,-3e-5],4e-5).field(x) if meta['shared_registration'] else np.zeros_like(x)
                values={'suite':suite,'case_id':r['case_id'],'method':r['method'],'physical_forecast_displacement_rmse_mm':[float(np.sqrt(np.mean((physical[:,j]-truth[:,j])**2))) for j in range(len(ids))],
                    'registered_noiseless_observation_rmse_mm':[float(np.sqrt(np.mean((pred[:,j]-truth[:,j]-true_bias)**2))) for j in range(len(ids))],
                    'strain_amplitude_rmse':abs(1-meta['R'])/2*r['forecast_strain_rmse'],
                    'stress_amplitude_rmse_n_mm2':abs(1-meta['R'])/2*r['forecast_stress_rmse_n_mm2'],
                    'metric_frame_corrected':bool('tx' in params),'parameters_predictions_unchanged':True}
                rows.append(values)
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'execution.json').write_text(json.dumps({'count':len(rows),'wall_seconds':time.perf_counter()-t,'fitted_registration_records':sum(r['metric_frame_corrected'] for r in rows),
        'original_record_files_modified':False,'changes':'score physical displacement and registered observations against corresponding truths; no new optimization or physics'},indent=2)+'\n')
    print('metric frame audit completed',len(rows),flush=True)
