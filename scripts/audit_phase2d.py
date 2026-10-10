"""Frozen gates and source/asset audit; failures remain part of the denominator."""
import argparse,json,hashlib,tomllib
from pathlib import Path
import numpy as np
from fatigue_dic_rl.evaluation.model_selection_metrics import audit_metrics,wilson
from fatigue_dic_rl.correction.robust_estimation import clean

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--config',default='configs/phase2d/study.toml');ap.add_argument('--output',required=True);a=ap.parse_args();r=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cfg=tomllib.loads(Path(a.config).read_text());rows=json.loads((r/'evaluation/records.json').read_text());plan=json.loads((r/'frozen_design/frozen_plan.json').read_text());expected=[x for x in plan['cases'] if x['split'] in ('test','ood','diagnostic')];assert len(rows)==len(expected)==160
    summary=audit_metrics(rows,cfg['qualification']);info=json.loads((r/'scientific_diagnostics/snr_records.json').read_text());summary['fixed_point_snr']=[dict(sigma_mm=s,rows=[dict(K=k,numerical_ranks=[q['numerical_rank'] for group in info for q in group if q['noise_sigma_mm']==s and q['K']==k],practical_ranks=[q['practical_rank'] for group in info for q in group if q['noise_sigma_mm']==s and q['K']==k],minimum_singular_values=[q['minimum_singular_value'] for group in info for q in group if q['noise_sigma_mm']==s and q['K']==k]) for k in (2,4,8)]) for s in cfg['study']['noise_mm']]
    baseline=json.loads((r/'baseline/source_manifest.json').read_text());changed=[p for p,h in baseline.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];summary['preexisting_source_changed']=changed;assert not changed
    inference=json.loads((r/'frozen_inference_manifest.json').read_text());changed_inference=[p for p,h in inference.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];summary['frozen_inference_changed']=changed_inference
    guard=json.loads((r/'deployment_guard_addendum.json').read_text());assert set(changed_inference)<=set([guard['file']]);assert hashlib.sha256(Path(guard['file']).read_bytes()).hexdigest()==guard['new_sha256']
    selected_unconverged=[x['case_id'] for x in rows if x['status']!='FAILED' and x['inference']['selection']['selected']!='B0' and x['inference']['candidates'][x['inference']['selection']['selected']].get('filtered',{}).get('status')!='CONVERGED']
    summary['deployment_guard_addendum']=dict(guard,selected_unconverged_cases=selected_unconverged,no_actual_decision_changed=not selected_unconverged);assert not selected_unconverged
    assets=json.loads((r/'baseline/historic_assets.json').read_text());changed_assets=[p for p,v in assets.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v];summary['historic_asset_changes']=changed_assets;assert not changed_assets
    weights=json.loads((r/'baseline/checkpoint_manifest.json').read_text());summary['historic_checkpoint_count']=len(weights);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in weights.items())
    full=json.loads((r/'pytest_final_v2_execution.json').read_text());summary['full_pytest']=full;summary['phase2c_core_replay']=json.loads((r/'phase2c_core_replay/replay.json').read_text());summary['historic_cli']=json.loads((r/'historic_regression/matrix.json').read_text());summary['temporal']=json.loads((r/'scientific_diagnostics/temporal/summary.json').read_text());summary['phase2b_regression']=json.loads((r/'phase2b_numeric_audit/summary.json').read_text());assert all(x['passed'] for x in summary['phase2b_regression'].values())
    summary['prefix_replay']=json.loads((r/'prefix_residual_replay/summary.json').read_text());assert summary['prefix_replay']['passed']
    summary['phase2c_primitives_regression']=json.loads((r/'phase2c_primitives_regression/execution.json').read_text());assert summary['phase2c_primitives_regression']['passed']
    summary['phase2c_seed_replay']=json.loads((r/'phase2c_seed_replay/execution.json').read_text());assert summary['phase2c_seed_replay']['passed']
    summary['scientific_followups']=json.loads((r/'scientific_followups_execution.json').read_text());assert all(x['exit_code']==0 for x in summary['scientific_followups'])
    summary['evaluation_execution']=json.loads((r/'evaluation/execution.json').read_text())
    summary['state_law_diagnostic']=json.loads((r/'state_law_diagnostic/summary.json').read_text())
    summary['frozen_parameter_steps']=json.loads((r/'frozen_parameter_steps/summary.json').read_text())
    summary['frozen_configuration_sha256']=hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    assert summary['frozen_configuration_sha256']==plan['config_sha256']
    summary['uncertainty_qualification']='NOT_QUALIFIED; no posterior calibration claim'
    boundary=[];tentative=[];costs=[];physical_extrema=[];max_fisher_relative_error=0.
    for file in (r/'evaluation/raw').glob('*/*/*_information.npz'):
        with np.load(file,allow_pickle=False) as data:
            j=data['J'];f=data['Fdata'];err=float(np.linalg.norm(f-j.T@j)/max(np.linalg.norm(f),1e-300));max_fisher_relative_error=max(max_fisher_relative_error,err);assert np.isfinite(j).all() and err<1e-12;assert np.linalg.eigvalsh(f).min()>=-1e-12*max(np.linalg.norm(f),1e-300)
    for file in (r/'evaluation/raw').glob('*/*_fields.npz'):
        with np.load(file,allow_pickle=False) as data:
            d=data['damage'];current=data['current_damage'];assert np.isfinite(d).all() and np.min(d)>=0 and np.max(d)<.95;assert np.min(np.diff(np.concatenate([current[:,None,:],d],axis=1),axis=1))>=-1e-12;assert np.isfinite(data['hazard']).all() and np.min(data['hazard'])>=0;physical_extrema.append([float(d.min()),float(d.max())])
    summary['field_audit']=dict(forecast_field_files=len(physical_extrema),damage_min=min(x[0] for x in physical_extrema),damage_max=max(x[1] for x in physical_extrema),fisher_reconstruction_max_relative_error=max_fisher_relative_error,irreversibility_violations=0)
    for row in rows:
        if row['status']=='FAILED':continue
        costs.append(row['inference']['cost'])
        for method,rec in row['inference']['candidates'].items():
            for stage in ('history','filtered'):
                fit=rec.get(stage,{})
                if fit.get('boundary'):boundary.append(dict(case_id=row['case_id'],method=method,stage=stage,parameters=fit['parameters']))
                if fit.get('status')=='UNCONVERGED_TENTATIVE':tentative.append(dict(case_id=row['case_id'],method=method,stage=stage))
    summary['boundary_solutions']=boundary;summary['unconverged_candidates']=tentative;summary['online_cost']=dict(case_total_seconds=[c['total_online_seconds'] for c in costs],selection_seconds=[c['selection_seconds'] for c in costs],forecast_seconds=[c['forecast_seconds'] for c in costs],fem_solves_four_candidates=[sum(c['candidate_fem_solves'].values()) for c in costs],gate_validation_fem_solves=[c['gate_validation_fem_solves'] for c in costs]);summary['full_high_noise_correction_observability']=all(x['history_K2_eligible']==20 for x in summary['noise']);summary['qualification']='PHASE2D_ROBUSTNESS_PARTIALLY_QUALIFIED';summary['RL']='NO-GO'
    summary['gate_values']={str(g):sum(x['inference']['shrinkage']['gate']==g for x in rows if x['status']!='FAILED') for g in (0.,.25,.5,.75,1.)}
    for name,g in summary['structural_gates'].items():
        g['improved_wilson95']=wilson(sum(x<1. for x in g['ratios']),g['count'])
    summary['main_family_selection_concordance']=[dict(family=family,case_count=len(group),physics=sum(x['inference']['selection']['selected']=='B0' for x in group),parameter=sum(x['inference']['selection']['selected'] in ('B1','B2') for x in group),correction=sum(x['inference']['selection']['selected']=='B5' for x in group)) for family in ('matched','parameter','S1') for group in [[x for x in rows if x['status']!='FAILED' and x['split']=='test' and x['family']==family]]]
    summary['qualification_limits']=dict(structural_selection_gate_passed=all(x['passed'] for x in summary['structural_gates'].values()),high_noise_observability_passed=summary['full_high_noise_correction_observability'],OOD_general_recovery_qualified=False,uncertainty_calibration_qualified=False)
    if full['exit_code']!=0 or not summary['phase2c_core_replay']['non_timing_fields_equal'] or any(x['returncode'] for x in summary['historic_cli']) or not summary['temporal']['passed'] or not all(x['passed'] for x in summary['matched_gates'].values()):summary['qualification']='PHASE2D_ROBUSTNESS_FAILED'
    (out/'audit.json').write_text(json.dumps(clean(summary),indent=2)+'\n');print(summary['qualification'],flush=True)
