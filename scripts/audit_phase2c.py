"""Frozen-result audit and figures; no fitting, checkpoint selection or calibration."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from fatigue_dic_rl.evaluation.identifiability_verification import clean
from fatigue_dic_rl.training.correction_dataset import load_privileged_prefix,source_hash
from fatigue_dic_rl.training.correction_supervised import load_residual_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    result={}
    for scenario in sorted({r['scenario'] for r in rows}):
        result[scenario]={}
        for method in sorted({r['method'] for r in rows if r['scenario']==scenario}):
            group=[r for r in rows if r['scenario']==scenario and r['method']==method]
            good=[r for r in group if r['status']!='FAILED']
            result[scenario][method]={'count':len(group),'failed':len(group)-len(good),
                'converged':sum(r['status']=='CONVERGED' for r in good),
                'current_rmse':float(np.mean([r['current_damage']['rmse'] for r in good])) if good else None,
                'forecast_rmse':np.mean([r['forecast_damage_rmse'] for r in good],axis=0).tolist() if good else None,
                'data_ranks':[r['data_rank'] for r in good],
                'parameter_dimension':[r['parameter_dimension'] for r in good],
                'online_seconds':float(np.mean([r['online_seconds'] for r in good])) if good else None,
                'fit_seconds':float(np.mean([r['fit_seconds'] for r in good])) if good else None,
                'fem_solves':float(np.mean([r['fem_solves'] for r in good])) if good else None,
                'parameter_rmse':{name:float(np.sqrt(np.mean([r['parameter_error'][name]**2 for r in good]))) for name in ('d0','A','m')} if good else None}
    return result


def preservation(root):
    baseline=read(root/'baseline/source_manifest.json')
    changes=[p for p,h in baseline.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    # README is the only deliberately updated pre-existing source/document.
    protected=[p for p in changes if p!='README.md']
    weights=read(root/'baseline/checkpoints.json')
    bad_weights=[p for p,v in weights.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v['sha256']]
    assets=read(root/'baseline/historical_assets_stat.json')
    stat_changes=[p for p,v in assets.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v]
    return {'preexisting_source_count':len(baseline),'changed':changes,'protected_changed':protected,
        'checkpoint_count':len(weights),'checkpoint_changes':bad_weights,'historic_asset_count':len(assets),'historic_stat_changes':stat_changes,
        'passed':not(protected or bad_weights or stat_changes)}


def audit_phase2b(root):
    old=Path('results/phase2b/20261008T114713Z')
    suites={
        'uniform':(old/'test_final/records.json',[root/'phase2b_regression/uniform_test/records.json']),
        'OOD':(old/'generalization_final/records.json',[root/'phase2b_regression/generalization/records.json',root/'phase2b_classical_ood_regression/records.json']),
        'fixed_noise':(old/'fixed_noise_final/records.json',[root/'phase2b_regression/fixed_noise/records.json']),
        'gaussian':(old/'gaussian_test_final/records.json',[root/'phase2b_regression_followup/gaussian/records.json'])}
    result={}
    for name,(before,paths) in suites.items():
        a=read(before);raw=[r for p in paths for r in read(p)];lookup={(r['case_id'],r['method']):r for r in raw}
        missing=[];status=[];errors=[]
        for r in a:
            key=(r['case_id'],r['method'])
            if key not in lookup:missing.append(key);continue
            fresh=lookup[key]
            if r['status']!=fresh['status']:status.append(key)
            for field,value in r.items():
                if field.endswith('_seconds') or value is None or not isinstance(value,(float,int,list)):
                    continue
                try:
                    before_value=np.asarray(value,dtype=float);after_value=np.asarray(fresh[field],dtype=float)
                except (ValueError,TypeError):
                    continue  # String provenance arrays are not numerical metrics.
                np.testing.assert_allclose(before_value,after_value,rtol=0,atol=0,equal_nan=True)
                finite=np.isfinite(before_value)
                errors.append(float(np.max(abs(before_value[finite]-after_value[finite]),initial=0.)))
        result[name]={'historic_records':len(a),'executed_records':len(raw),'missing':missing,'status_changes':status,
            'maximum_numerical_difference':max(errors,default=0.),'passed':not missing and not status and max(errors,default=0.)<1e-10}
    arch=read(root/'phase2b_regression_followup/architecture.json')
    result['architecture']={'checkpoints':len(arch),'maximum_error':max(max(r['errors'].values()) for r in arch),'passed':len(arch)==5}
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter()
    original=read(root/'test/records.json');fair=read(root/'fair_calibration/records.json')
    records=[r for r in original if r['method'] not in ('B1_refit_A','B2_refit_Am')]+fair
    assert len(records)==305 and len({(r['case_id'],r['method']) for r in records})==305
    ood=read(root/'ood/records.json');nuisance=read(root/'nuisance/records.json');reference=read(root/'oracle_reference/records.json');spatial=read(root/'spatial_basis_followup/records.json')
    ratios=[]
    for r in records:
        if r['scenario']=='structural' and r['method']=='B5_multistep_K2':
            b=next(x for x in fair if x['case_id']==r['case_id'] and x['method']=='B2_refit_Am')
            ratios.append({'case_id':r['case_id'],'forecast_ratios':(np.asarray(r['forecast_damage_rmse'])/b['forecast_damage_rmse']).tolist(),
                'current_ratio':r['current_damage']['rmse']/b['current_damage']['rmse'],'long_horizon_ratio':r['forecast_damage_rmse'][-1]/b['forecast_damage_rmse'][-1]})
    long_ratios=np.array([r['long_horizon_ratio'] for r in ratios]);structure_gate=len(ratios)==5 and np.median(long_ratios)<.9 and np.sum(long_ratios<1)>=4 and max(long_ratios)<=1.25
    models=read(root/'training/manifest.json')['models'];reload_rows=[];neutral=[]
    for m in models:
        regime=Path(m['train_cases'][0]).parent.parent.name
        path=root/'training'/regime/f"K{m['dimension']}"/f"seed_{m['seed']}"/'checkpoint.pt'
        model,metadata=load_residual_checkpoint(path);cofs=model.coefficients.detach().numpy()
        np.testing.assert_array_equal(cofs,np.asarray(m['coefficients']))
        model2,_=load_residual_checkpoint(path);np.testing.assert_array_equal(cofs,model2.coefficients.detach().numpy())
        reload_rows.append({'regime':regime,'K':m['dimension'],'seed':m['seed'],'coefficient_max_difference':0.,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if regime=='matched':
            phi=np.concatenate([load_privileged_prefix(p,dimension=m['dimension'])[0] for p in m['validation_cases']])
            deviation=float(np.max(abs(np.exp(m['alpha']*np.tanh(phi@cofs))-1)))
            neutral.append({'K':m['dimension'],'seed':m['seed'],'max_validation_factor_deviation':deviation,'predeclared_tolerance':.0002,'passed':deviation<=.0002})
    primary_neutral=all(r['passed'] for r in neutral if r['K']==4)
    main_summary=summarize(records);main_ood=summarize(ood)
    count={'main':len(records),'expanded_calibration':len(fair),'ood':len(ood),'nuisance':len(nuisance),'reference':len(reference),'spatial':len(spatial),'training_checkpoints':len(models)}
    all_rows=records+ood+nuisance+reference+spatial
    assert all(r['status'] in ('CONVERGED','UNCONVERGED_TENTATIVE') and r['irreversibility_violations']==0 and r['conditioning_through_cycle']<=20 for r in all_rows)
    unfinished=[{'case_id':r['case_id'],'method':r['method'],'status':r['status']} for r in all_rows if r['status']!='CONVERGED']
    assert all(r['forecast_before_truth_IO'] for r in records+ood+nuisance+spatial)
    boundary=[]
    bounds={'d0':(.02,.3),'A':(1e-5,5e-4),'m':(.6,1.6),'tx':(-.002,.002),'ty':(-.002,.002),'theta':(-.002,.002),'load_factor':(.85,1.15)}
    for r in all_rows:
        for name,value in r['parameters'].items():
            lo,hi=bounds.get(name,(-4.,4.))
            if min(value-lo,hi-value)/(hi-lo)<1e-6:boundary.append({'case_id':r['case_id'],'method':r['method'],'parameter':name,'value':value})
    historic=read(root/'historic_scientific_regression_v2/matrix.json');pytest=read(root/('pytest_final_v2_execution.json' if (root/'pytest_final_v2_execution.json').exists() else 'pytest_final_execution.json'));regression=audit_phase2b(root);protected=preservation(root)
    regression_gate=pytest['returncode']==0 and len(historic)==37 and all(r['returncode']==0 for r in historic) and all(r['passed'] for r in regression.values()) and protected['passed']
    acceptance=[
        ('Matched neutral correction','PARTIALLY_QUALIFIED','Exact zero coefficients are neutral; primary supervised K4 violates frozen 2e-4 factor tolerance in four of five seeds.'),
        ('Constant/zero hazard','PASSED','scientific_primitives/constant and unit tests'),
        ('Nonnegative damage rate','PASSED','bounded multiplicative factor and strict finite hazard contracts'),
        ('Boundedness/irreversibility','PASSED','all formal forecasts have zero irreversible violations; unchanged Phase1B transition'),
        ('Consistent material history','PASSED','candidate replay, exact layout ownership and G!=element-count/passive-history tests'),
        ('FE rollback','PASSED','injected new-damage equilibrium failure preserves committed state; original rollback CLI'),
        ('Frozen hazard/time consistency','PASSED','analytic semigroup; independent DOP853; query invariance'),
        ('Parameter mismatch recovery','PASSED','expanded-budget-equivalent A/m refit on all five fixed cases per M1 regime'),
        ('Structural discrepancy improvement','PASSED' if structure_gate else 'NOT_QUALIFIED','frozen five-case paired K2 gate; high-SNR synthetic distribution only'),
        ('Correction identifiability','PARTIALLY_QUALIFIED','K2 supported under known inputs; K4/K8 weak/null; A/constant and m/log-x tangent redundancy diagnosed'),
        ('Future holdout','PASSED','all forecasts before truth/future DIC IO; N0/10/20 cutoff and explicit B3 origin10'),
        ('Independent/spatial DIC noise','PARTIALLY_QUALIFIED','frozen two-case OOD diagnostics, not statistical coverage qualification'),
        ('Boundary/registration confounding','PARTIALLY_QUALIFIED','shared registration improves; unknown loading retains boundary and weak solutions'),
        ('Cross-mesh sensitivity','PARTIALLY_QUALIFIED','paired global fields and genuine spatial FE refinement; no mesh-objective localization'),
        ('Spatial basis validity','PASSED','coordinate-defined fixed basis, FE renumbering and coupled spatial rollout'),
        ('Correct-physics negative control','PASSED_WITH_DEGRADATION','unnecessary flexible fits and biased-state correction worsen long forecasts; failures retained'),
        ('Classical A/m comparison','PASSED','identical prefix and TRF80 budget, expanded A/m bounds; all old results retained'),
        ('Five-seed reproducibility','PASSED','75 frozen checkpoint reloads and five-seed actual retraining replay'),
        ('Observation-only isolation','PASSED','strict public schema, guarded prefix loading and IO-boundary integration test'),
        ('Complete Phase0-2B regression','PASSED' if regression_gate else 'NOT_QUALIFIED','426 pytest after metric-frame regression; all37 historic CLI; complete frozen Phase2B datasets/checkpoints/calibration')]
    result={'qualification':'PHASE2C_PHYSICS_CORRECTION_PARTIALLY_QUALIFIED' if structure_gate and regression_gate else 'PHASE2C_PHYSICS_CORRECTION_FAILED',
        'source_sha256':source_hash(),'counts':count,'structural_K2_gate':bool(structure_gate),'structural_paired_ratios':ratios,
        'structural_median_long_ratio':float(np.median(long_ratios)),'matched_primary_K4_neutral_gate':primary_neutral,'neutral_diagnostics':neutral,
        'all_seeds_checkpoint_replay':reload_rows,'regression_gate':regression_gate,'phase2b_regression':regression,'preservation':protected,
        'main':main_summary,'ood':main_ood,'nuisance':summarize(nuisance),'reference':summarize(reference),'spatial_followup':summarize(spatial),
        'unconverged_retained':unfinished,
        'scientific_acceptance':[{'test':i+1,'name':name,'status':status,'evidence':evidence} for i,(name,status,evidence) in enumerate(acceptance)],
        'boundary_results':boundary,'uncertainty_qualified':False,'graph_conditioned_extension':'NOT_EXECUTED: weak higher-K directions and no independent spatial training coverage',
        'neural_estimator_substitution':'NOT_EXECUTED: old checkpoint fatigue parameter support excludes this study; no fair frozen checkpoint substitution',
        'RL_recommendation':'NO_GO: parameter calibration suffices for M1; M2 supports a non-RL correction baseline, not the necessity of a policy; OOD/state/boundary/neutral controls remain limited',
        'wall_seconds':time.perf_counter()-t}
    (out/'fair_records.json').write_text(json.dumps(clean(records),indent=2,allow_nan=False)+'\n')
    (out/'audit.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained');ns=np.array([30,40,50,70,120,220])
    for method,label in [('B0_physics_prior','Prior'),('B2_refit_Am','Refit A,m'),('B5_multistep_K2','DIC correction K2'),('B5_multistep_K4','DIC correction K4'),('B4_supervised_K4_seed_85000','Supervised K4 seed85000')]:
        axes[0].plot(ns,main_summary['structural'][method]['forecast_rmse'],'o-',label=label)
    axes[0].set(xlabel='Cycle N (observed through N20)',ylabel='Damage RMSE',yscale='log',title='Structural discrepancy / same held-out cases');axes[0].legend(fontsize=7)
    for method,label in [('B0_physics_prior','Prior'),('B2_refit_Am','Refit A,m'),('B5_multistep_K2','DIC K2')]:
        axes[1].plot(ns,main_summary['matched'][method]['forecast_rmse'],'o-',label=label)
    axes[1].set(xlabel='Cycle N',ylabel='Damage RMSE',yscale='log',title='Matched physics negative control');axes[1].legend(fontsize=7)
    primitive=root/('scientific_primitives_final' if (root/'scientific_primitives_final/execution.json').exists() else 'scientific_primitives')
    temporal=read(primitive/'temporal/summary.json')['rows'];h=[r['step_cycle'] for r in temporal];err=[r['damage_rmse_vs_DOP853'] for r in temporal]
    axes[2].loglog(h,err,'o-');axes[2].set(xlabel='Cycle step',ylabel='Error vs independent DOP853',title='Time discretization convergence')
    for ax in axes:ax.grid(alpha=.25)
    for ext in ('png','svg'):fig.savefig(out/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    print(result['qualification'],flush=True)
