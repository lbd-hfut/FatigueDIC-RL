"""All-world paired statistics; ABSTAIN and failures cannot improve denominators."""
import numpy as np
from fatigue_dic_rl.evaluation.model_selection_metrics import wilson,paired_mean_interval


def summarize_worlds(worlds,config):
    strategies=config['design']['strategies'];result={};q=config['qualification'];seed=config['statistics']['bootstrap_seed'];reps=config['statistics']['bootstrap_repeats']
    for strategy in strategies:
        finished=[r for r in worlds if r['arms'].get(strategy,{}).get('status')=='COMPLETED'];matched=[r for r in worlds if r['evaluation_only_family']=='matched' and not r.get('options') and r.get('noise_model')=='independent'];structural=[r for r in worlds if r['evaluation_only_family']=='S1' and not r.get('options') and r.get('noise_model')=='independent'];gates={}
        excess=[];activated=0;matched_failures=0
        for r in matched:
            a=r['arms'].get(strategy,{})
            if a.get('status')!='COMPLETED':matched_failures+=1;continue
            excess.append(a['methods']['E9']['forecast_damage_rmse'][-1]-a['methods']['B0']['forecast_damage_rmse'][-1]);activated+=a['inference']['E9']['deployed_model']=='B5'
        stat=paired_mean_interval(excess,seed=seed,repeats=reps);wi=wilson(activated,len(matched));gates['matched']=dict(**stat,independent_worlds=len(matched),failures=matched_failures,false_K2=activated,false_K2_wilson95=wi,passed=bool(not matched_failures and excess and stat['bootstrap95'][1]<=q['matched_mean_excess_upper95'] and stat['maximum']<=q['matched_max_excess'] and wi[1]<=q['matched_false_K2_Wilson_upper95']))
        ratios=[];gains=[];improved=0;structural_failed=0
        for r in structural:
            a=r['arms'].get(strategy,{})
            if a.get('status')!='COMPLETED':structural_failed+=1;continue
            e=a['methods']['E9']['forecast_damage_rmse'][-1];b=a['methods']['B7']['forecast_damage_rmse'][-1];ratios.append(e/max(b,1e-15));gain=b-e;gains.append(gain);improved+=gain>=q['minimum_absolute_gain']
        gates['structural']=dict(independent_worlds=len(structural),evaluated=len(ratios),failures=structural_failed,ratios_E9_vs_same_design_B7=ratios,median_ratio=float(np.median(ratios)) if ratios else None,practical_improvements=improved,improved_fraction=improved/len(structural) if structural else None,wilson95=wilson(improved,len(structural)),paired_gain=paired_mean_interval(gains,seed=seed,repeats=reps),passed=bool(structural and not structural_failed and improved/len(structural)>=q['structural_improvement_fraction'] and np.median(ratios)<=q['structural_median_ratio_vs_equal_design_B7']))
        groups=[]
        for family in sorted({r['evaluation_only_family'] for r in worlds}):
            rows=[r for r in finished if r['evaluation_only_family']==family];groups.append(dict(family=family,physical_worlds=len([r for r in worlds if r['evaluation_only_family']==family]),completed=len(rows),mean_N220_damage_rmse={n:float(np.mean([r['arms'][strategy]['methods'][n]['forecast_damage_rmse'][-1] for r in rows if n in r['arms'][strategy]['methods']])) if rows else None for n in ('B0','B2','B5','B7','E9')},E9_decisions={s:sum(r['arms'][strategy]['inference']['E9']['decision']==s for r in rows) for s in ('ABSTAIN','KEEP_PHYSICS','CORRECTION_EVIDENCE')}))
        result[strategy]=dict(physical_worlds=len(worlds),completed=len(finished),failures=len(worlds)-len(finished),abstentions=sum(r['arms'][strategy]['inference']['E9']['decision']=='ABSTAIN' for r in finished),gates=gates,families=groups,paired_E9_minus_same_design_B7=paired_mean_interval([r['arms'][strategy]['methods']['E9']['forecast_damage_rmse'][-1]-r['arms'][strategy]['methods']['B7']['forecast_damage_rmse'][-1] for r in finished],seed=seed,repeats=reps),median_arm_seconds=float(np.median([r['arms'][strategy]['cost']['arm_seconds'] for r in finished])) if finished else None)
    return dict(independent_worlds=len(worlds),paired_design_arms_not_independent_worlds=True,strategies=result,qualification='PHASE2E_OBSERVABILITY_QUALIFIED' if all(result[s]['gates']['matched']['passed'] and result[s]['gates']['structural']['passed'] and result[s]['failures']==0 for s in ('fisher','discrimination')) else 'PHASE2E_OBSERVABILITY_PARTIALLY_QUALIFIED',RL='NO-GO',interval_notes='Wilson binomial intervals and prespecified within-world paired mean bootstrap; small family/condition subgroups remain exploratory')
