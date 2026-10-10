"""Case-level paired uncertainty and frozen qualification gates (no tuning)."""
import numpy as np


def wilson(success,total,z=1.959963984540054):
    if total==0:return [None,None]
    p=success/total;d=1+z*z/total;mid=(p+z*z/(2*total))/d;half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [float(mid-half),float(mid+half)]


def paired_mean_interval(values,*,seed=94991,repeats=10000):
    x=np.asarray(values,float)
    if not len(x):return dict(mean=None,bootstrap95=[None,None],count=0)
    rng=np.random.default_rng(seed);means=x[rng.integers(0,len(x),size=(repeats,len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()),bootstrap95=np.quantile(means,[.025,.975]).tolist(),count=len(x),maximum=float(x.max()))


def audit_metrics(records,qualification):
    good=[r for r in records if r['status']!='FAILED'];main=[r for r in good if r['split']=='test'];matched=[r for r in main if r['family']=='matched'];gates={}
    for name in ('B7','B8'):
        excess=[r['methods'][name]['forecast_damage_rmse'][-1]-r['methods']['B0']['forecast_damage_rmse'][-1] for r in matched if name in r['methods']]
        stat=paired_mean_interval(excess);activation=sum(r['inference']['selection']['selected']=='B5' if name=='B7' else r['inference']['shrinkage']['gate']>0 for r in matched);wi=wilson(activation,len(matched));passed=bool(len(excess)==40 and stat['bootstrap95'][1]<=qualification['matched_mean_excess_upper95'] and stat['maximum']<=qualification['matched_max_excess'] and wi[1]<=qualification['matched_false_correction_wilson_upper95'])
        gates[name]=dict(**stat,false_correction_activations=activation,false_activation_wilson95=wi,passed=passed)
    param=[r for r in main if r['family']=='parameter' and r['inference']['selection']['eligibility'].get('B2')];preferred=sum(r['inference']['selection']['selected'] in ('B1','B2') for r in param)
    preference=dict(count=len(param),preferred=preferred,fraction=preferred/len(param) if param else None,wilson95=wilson(preferred,len(param)),passed=bool(param and preferred/len(param)>=qualification['parameter_model_preference_identifiable_min']))
    structural=[r for r in main if r['family']=='S1' and r['inference']['selection']['eligibility'].get('B5') and 'B2' in r['methods']]
    structural_gates={}
    for name in ('B7','B8'):
        ratios=[r['methods'][name]['forecast_damage_rmse'][-1]/max(r['methods']['B2']['forecast_damage_rmse'][-1],1e-15) for r in structural];improved=sum(x<1 for x in ratios)
        structural_gates[name]=dict(count=len(ratios),ratios=ratios,median=float(np.median(ratios)) if ratios else None,improved_fraction=improved/len(ratios) if ratios else None,passed=bool(ratios and np.median(ratios)<=qualification['structural_identifiable_median_ratio_vs_B2_max'] and improved/len(ratios)>=qualification['structural_identifiable_improved_fraction_min']))
    noise=[]
    for sigma in sorted({r['sigma_mm'] for r in main}):
        group=[r for r in main if r['sigma_mm']==sigma];ms={}
        for name in ('B0','B1','B2','B5','B7','B8'):
            cases=[r for r in group if name in r['methods']];ms[name]=dict(count=len(cases),damage_rmse_mean=np.mean([r['methods'][name]['forecast_damage_rmse'] for r in cases],axis=0).tolist() if cases else None)
        noise.append(dict(sigma_mm=sigma,independent_cases=len(group),selected_counts={n:sum(r['inference']['selection']['selected']==n for r in group) for n in ('B0','B1','B2','B5')},history_K2_eligible=sum(r['inference']['selection']['eligibility'].get('B5',False) for r in group),methods=ms))
    return dict(total=len(records),failed=sum(r['status']=='FAILED' for r in records),candidate_failures=sum(len(r['inference']['cost']['candidate_failures']) for r in good),matched_gates=gates,parameter_preference=preference,structural_gates=structural_gates,noise=noise,statistics='independent physical cases; load specimens, frames and DIC points are not extra replicates; paired within-case model errors')
