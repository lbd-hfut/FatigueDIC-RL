"""Independent arithmetic and provenance review of finished Development artifacts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from fatigue_dic_rl.correction.robust_estimation import clean


def run(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());diag=json.loads((root/'diagnostics/summary.json').read_text());records=[];errors=[];difference=0.;violations=0;upper=0.
    for r in summary['results']:
        if r['status']!='COMPLETED':errors.append(dict(case=r['case_id'],reason=r['status']));continue
        out=root/'inference'/r['case_id'];truefile=out/'evaluation_truth_after_forecast.npz'
        with np.load(truefile) as z:truth=dict(z)
        w=truth['weights']/truth['weights'].sum()
        for m,s in r['scores'].items():
            file=out/(m+'_forecast_before_truth.npz')
            with np.load(file) as z:p=dict(z)
            d=p['damage']-truth['damage'][:,1:];dc=p['current_damage']-truth['damage'][:,0]
            by=np.sqrt(np.mean(np.sum(d*d*w,axis=-1),axis=0));overall=float(np.sqrt(np.mean(np.sum(d*d*w,axis=-1))));cur=float(np.sqrt(np.mean(np.sum(dc*dc*w,axis=-1))));u=float(np.sqrt(np.mean((p['u']-truth['u'][:,1:])**2)))
            difference=max(difference,float(np.max(np.abs(by-np.array(s['damage_RMSE_by_horizon'])))),abs(overall-s['damage_RMSE']),abs(cur-s['current_damage_RMSE']),abs(u-s['heldout_displacement_RMSE_mm']))
            v=int(np.sum(np.diff(np.concatenate([p['current_damage'][:,None],p['damage']],axis=1),axis=1)<-1e-12));violations+=v;upper=max(upper,float(p['damage'].max()))
            before=file.stat().st_mtime_ns<truefile.stat().st_mtime_ns
            if not before or v or np.min(p['damage'])<0 or np.max(p['damage'])>=.95:errors.append(dict(case=r['case_id'],method=m,reason='chronology or physical bounds'))
            records.append(dict(case_id=r['case_id'],method=m,forecasts_before_truth=before,cycles=p['cycles'].tolist(),independent_RMSE_recomputed=True))
    # Fixed-point independent observations: additional data Fisher must be PSD.
    psd={}
    for m in ('B2','B5'):
        with np.load(root/'inference'/'p2f-dev-002-multi_long'/m/'full_information.npz') as z:coarse=z['Fdata']
        with np.load(root/'diagnostics'/'dense_history'/(m+'.npz')) as z:dense=z['F_data']
        ev=np.linalg.eigvalsh(dense-coarse);psd[m]=dict(eigenvalues_increment=ev,fixed_linearization_point=True,extra_observation_cost=640)
        if ev.min() < -1e-5:errors.append(dict(method=m,reason='fixed-point cumulative Fisher not PSD'))
    rules=json.loads((root/'plan.json').read_text())['frozen_rules'];changed=[p for p,h in rules.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    if changed:errors.append(dict(reason='frozen rules changed',paths=changed))
    out=dict(scope='DEVELOPMENT evidence only; no independent qualification',independent_worlds=3,full_protocols=10,information_only_protocols=3,forecast_artifacts=len(records),maximum_metric_recomputation_difference=difference,irreversibility_violations=violations,maximum_predicted_damage=upper,maximum_true_damage=max(r['max_true_damage'] for r in summary['results'] if r['status']=='COMPLETED'),fixed_point_Fisher_PSD=psd,frozen_rules_unchanged=not changed,records=records,errors=errors,passed=not errors and difference<1e-12,uncertainty_calibrated=False,profile_scope='predeclared bounded-prefix fits followed by descriptive forecast scoring; not a global confidence region',formal_independent_tests=False)
    (root/'scientific_audit.json').write_text(json.dumps(clean(out),indent=2)+'\n');print(json.dumps(clean({k:v for k,v in out.items() if k!='records'})))
    if not out['passed']:raise ValueError('scientific artifact audit failed; retain all failures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',required=True);a=p.parse_args();run(a.development)
