"""Independent time reference and fixed-point SNR diagnostics, outside estimation."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter
from fatigue_dic_rl.correction.physics_predictor import CorrectionPredictor
from fatigue_dic_rl.correction.law_family_benchmarks import ReferenceLawSpecification,FAMILIES
from fatigue_dic_rl.correction.correction_gate import GatedLawSpecification
from fatigue_dic_rl.correction.robust_estimation import clean
from fatigue_dic_rl.correction.identifiability_gate import observability
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian,posterior_uncertainty
from fatigue_dic_rl.inverse.parameterizations import UniformDamage
from fatigue_dic_rl.fatigue import MaterialHistoryState,EnergyAmplitudeDriver
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from .correction_metrics import damage_metrics


def temporal_law_families(case,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);a=load_correction_acquisition(case);p=a.predictors[3];space=p.parameter_space;z=space.encode(dict(d0=.12));layout=p.pattern.quadrature_layout;driver=EnergyAmplitudeDriver();backend=Tri3MechanicsSolver();rows=[]
    settings=[(f,ReferenceLawSpecification(f,A=1.28e-4 if f=='parameter' else 8e-5,m=1.18 if f=='parameter' else 1.,strength={'S2':.5,'S3':.55,'S5':.3}.get(f,3.))) for f in FAMILIES]
    settings += [('K2_gate_'+str(g),GatedLawSpecification(g)) for g in (0.,.5,1.)]
    for name,spec in settings:
        hist=p.load_history
        if name=='S3':hist=tuple(replace(c,load_min=.3*c.load_max,nominal_stress_ratio=.3) for c in hist)
        physical=dict(A=8e-5,m=1.,c0=.5,c1=-.1);law=spec.build(physical,np.zeros(layout.count,int));initial=np.full(layout.count,.12);calls=0
        for seg in hist:
            def rhs(n,d):
                nonlocal calls
                state=MaterialHistoryState.initialize(layout,layout.field(d,'1'),damage_max=.95,cycle_count=n);cyc=replace(seg,N_start=float(n),delta_N=1.)
                lo=backend.solve(cyc.request(p.pattern,state.damage,n,'min'));hi=backend.solve(cyc.request(p.pattern,state.damage,n,'max'));calls+=2
                return (.95-d)*law.hazard(driver.evaluate(lo,hi),cyc,state).values
            reference=solve_ivp(rhs,(seg.N_start,seg.N_end),initial,method='DOP853',rtol=1e-10,atol=1e-12)
            if not reference.success:raise RuntimeError(reference.message)
            initial=reference.y[:,-1]
        if name.startswith('K2_gate'):
            from fatigue_dic_rl.correction.acquisition import parameter_space
            sp=parameter_space(mode='hybrid',coefficients=[.5,-.1]);zz=sp.encode(dict(d0=.12))
        else:sp,zz=space,z
        steps=[];fields=dict(reference=initial,quadrature_coordinates_mm=layout.coordinates,physical_weights_mm2=layout.physical_weights)
        for dt in (4.,2.,1.,.5):
            pred=CorrectionPredictor(UniformDamage(),sp,p.pattern,law_spec=spec,load_history=hist,cycle_step=dt);s=pred.damage_states(zz,[220.])[220.];e=damage_metrics(s.damage.values,initial,layout.physical_weights)['rmse'];extra=pred.damage_states(zz,[0.,7.,20.,33.,220.])[220.]
            gap=float(np.max(abs(s.damage.values-extra.damage.values)));assert gap==0.
            steps.append(dict(step_cycles=dt,damage_rmse_vs_DOP853=e,query_difference=gap,damage_min=float(s.damage.values.min()),damage_max=float(s.damage.values.max())));fields['step_'+str(dt)]=s.damage.values
        for x,y in zip(steps,steps[1:]):y['order']=float(np.log2(x['damage_rmse_vs_DOP853']/y['damage_rmse_vs_DOP853'])) if y['damage_rmse_vs_DOP853']>0 else None
        passed=all(y['damage_rmse_vs_DOP853']<x['damage_rmse_vs_DOP853'] for x,y in zip(steps,steps[1:])) and steps[-1]['order'] is not None and .8<steps[-1]['order']<1.2
        rows.append(dict(law=name,steps=steps,reference_fem_solves=calls,passed=passed));np.savez_compressed(out/(name+'_fields.npz'),**fields)
    (out/'summary.json').write_text(json.dumps(clean(dict(rows=rows,passed=all(r['passed'] for r in rows))),indent=2)+'\n');return rows


def snr_probe_task(task):
    folder,output=task;out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=[]
    # Same public nominal linearization for each K; no private state or labels.
    for k in (2,4,8):
        a=load_correction_acquisition(folder,mode='hybrid',dimension=k);adapter=MultiTrajectoryAdapter(a.predictors,a.batches);z=adapter.parameter_space.encode(dict(d0=.12,**{f'c{i}':0. for i in range(k)}));s,method=checked_local_jacobian(adapter.whitened_prediction,z);u=posterior_uncertainty(s,adapter.parameter_space.scales,normalized=z,derivative_method=method);rows.append(dict(K=k,**observability(u,z),singular_values=u.singular_values_data,parameter_names=adapter.parameter_space.names,noise_sigma_mm=a.metadata['sigma_mm'],case_id=a.metadata['case_id']))
        np.savez_compressed(out/f'K{k}_information.npz',J=u.jacobian_whitened,F=u.data_fisher,singular_values=u.singular_values_data,weak=u.weak_directions,null=u.data_null_directions)
    (out/'summary.json').write_text(json.dumps(clean(rows),indent=2)+'\n');return clean(rows)
