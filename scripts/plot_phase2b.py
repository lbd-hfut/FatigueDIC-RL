"""Export standard scientific figures from frozen raw summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);summary=json.loads((root/'test_final/summary.json').read_text());fair=summary['fair_comparison_first_frozen_cases'];all_methods=summary['methods'];groups=['physics_prior','independent_frame_inverse','prefix_cold_map','sequential_warm_map','graph_initial','graph_gru','physics_global','physics_shape','physics_shape_uncertainty'];names=['Prior','Frame inverse','Cold MAP','Warm MAP','Graph N0','Graph GRU','Physics global','Physics shape','Shape+cov']
    def values(group,field,methods=fair):
        keys=[k for k in methods if k==group or k.startswith(group+'_seed_')];return [methods[k][field] for k in keys if methods[k].get(field) is not None]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,field,title in zip(axs,['current_damage_rmse_mean','online_seconds_mean'],['Current damage RMSE (same16 cases)','Online pipeline seconds (same16 cases)']):
        for i,g in enumerate(groups):
            v=values(g,field)
            if v:ax.scatter(np.full(len(v),i),v,s=22);ax.plot([i-.2,i+.2],[np.mean(v)]*2,color='black')
        ax.set_xticks(range(len(groups)),names,rotation=45,ha='right');ax.set_title(title);ax.grid(alpha=.2)
        if 'seconds' in field:ax.set_yscale('log')
    save(fig,'fair_accuracy_cost')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for j,ax in enumerate(axs):
        for i,g in enumerate(groups):
            v=values(g,'forecast_damage_rmse_mean');v=[r[j] for r in v]
            if v:ax.scatter(np.full(len(v),i),v,s=22);ax.plot([i-.2,i+.2],[np.mean(v)]*2,color='black')
        ax.set_xticks(range(len(groups)),names,rotation=45,ha='right');ax.set_title(f'Open-loop D{[30,40][j]} RMSE (same16 cases)');ax.grid(alpha=.2)
    save(fig,'open_loop_forecasts')
    keys=sorted(k for k in all_methods if k.startswith('physics_shape_uncertainty'))
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for j,ax in enumerate(axs):
        raw=[all_methods[k]['raw_parameters']['coverage'][j]['coverage'] for k in keys];cal=[all_methods[k]['parameters']['coverage'][j]['coverage'] for k in keys];x=np.arange(len(keys));ax.plot(x,raw,'o-',label='Raw learned');ax.plot(x,cal,'s-',label='Calibration split scaled');ax.axhline(.95,color='black',ls='--',label='Nominal .95');ax.set_ylim(0,1.01);ax.set_xticks(x,[k.rsplit('_',1)[1] for k in keys]);ax.set_title(['d0','Fatigue A'][j]+' test coverage (128 cases/seed)');ax.set_xlabel('Training seed');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'parameter_calibration')
    ood=json.loads((root/'generalization_final/summary.json').read_text())['scenarios'];fig,ax=plt.subplots(figsize=(13,4));labels=[];means=[]
    for name,methods in ood.items():
        v=[s['forecast_damage_rmse_mean'][1] for k,s in methods.items() if k.startswith('physics_shape_uncertainty') and s.get('forecast_damage_rmse_mean') is not None]
        labels.append(name);means.append(np.mean(v) if v else np.nan)
    ax.bar(np.arange(len(labels)),means);ax.set_xticks(np.arange(len(labels)),labels,rotation=55,ha='right');ax.set_ylabel('Mean D40 RMSE over5 training seeds');ax.set_title('Frozen OOD cases (12/scenario), ID calibration unchanged');ax.grid(axis='y',alpha=.2);save(fig,'OOD_forecast_diagnostics')
    (out/'manifest.json').write_text(json.dumps({'source_summaries':['test_final/summary.json','generalization_final/summary.json'],'aggregation':'every training seed retained; bars/black lines mean all seeds, no best seed selection','files':[p.name for p in sorted(out.iterdir())]},indent=2)+'\n')
