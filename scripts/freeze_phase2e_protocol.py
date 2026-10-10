"""Freeze independent worlds only AFTER the separate development/validation run."""
import argparse,json,hashlib,tomllib
from datetime import datetime,timezone
from pathlib import Path
from verify_phase2e import cases
from fatigue_dic_rl.correction.design_provenance import runtime_manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--development',required=True);ap.add_argument('--validation',required=True);ap.add_argument('--runtime-validation',required=True);ap.add_argument('--output',required=True);ap.add_argument('--config',default='configs/phase2e/study.toml');a=ap.parse_args();evidence={}
    for split,p in [('development',a.development),('validation',a.validation)]:
        path=Path(p)/'results.json';r=json.loads(path.read_text());expected={c['case_id'] for c in cases(split)}
        if {c['case_id'] for c in r['worlds']}!=expected:raise ValueError('all prespecified development/validation worlds must finish, including failures')
        evidence[split]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_worlds=len(r['worlds']),failed_worlds=sum(c['status']=='FAILED' for c in r['worlds']))
    config=Path(a.config);c=tomllib.loads(config.read_text());manifest=runtime_manifest()
    v=Path(a.runtime_validation);run=json.loads((v/'results.json').read_text());execution=json.loads((v/'execution_plan.json').read_text())
    if execution['runtime_manifest']!=manifest or any(r['status']!='COMPLETED' for r in run['worlds']):raise ValueError('a complete smoke validation of the exact final runtime is required before freeze')
    evidence['final_runtime_validation']=dict(path=str((v/'results.json').resolve()),sha256=hashlib.sha256((v/'results.json').read_bytes()).hexdigest(),counted_as_new_independent_worlds=False)
    record=dict(created_UTC=datetime.now(timezone.utc).isoformat(),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),runtime_manifest=manifest,test_cases=cases('test'),evidence=evidence,rules=c,primary_strata=dict(matched='test000..007',parameter='test008..015',structural='test016..023',OOD='test024..031',uncertain_observation='test032..039'),measurement_budget=dict(shared_N0_N10_scalar_DIC=320,new_N20_cost_units=160,primary_new_DIC_components=160,reaction_secondary_new_DIC_components=150,reaction_secondary_components=5,reaction_cost_units=2),loading_budget=dict(new_cycles_per_specimen=10.,specimens_per_world=5,nominal_stress_bound_n_mm2=35.,max_damage=.6,amplitude_and_R_exposure_different=True,not_equal_work_budget=True),computational_budget=dict(max_nfev=80,public_virtual_conditions=8,primary_designs=5,maximum_test_workers=6,exact_realized_costs_reported=True,designer_library_cost_shared_for_paired_benchmark=True),statistical_plan=dict(unit='physical world; all specimens/frames/designs paired',test_worlds=40,Wilson_overall_worst_half_width_approximately=.148,primary_stratum_worlds=8,OOD_worlds_per_family=2,uncertain_worlds_per_condition=2,precision_reason='small CPU direction screen; overall Wilson precision about +/-15 percentage points. Negative-control upper10% cannot be certified by n8 even with zero activations.',ABSTAIN='retained in all denominators; operational fallback B0 or accepted simple calibration',failure='retained, counts as no practical improvement; no post-outcome exclusions',benefit='prespecified absolute N220 RMSE gain >=.001 and overall family gates in config; report all shorter horizons',no_further_test_tuning=True,old_phase2d_worlds_qualification_count=0),scope='non-RL, low-dimensional classical inference; conditional Gaussian geometry is not calibrated model probability')
    out=Path(a.output)
    with out.open('x') as f:json.dump(record,f,indent=2)
    print('independent protocol frozen:',out,'physical worlds',len(record['test_cases']))
