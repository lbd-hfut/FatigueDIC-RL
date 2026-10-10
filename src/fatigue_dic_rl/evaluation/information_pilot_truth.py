"""Privileged synthetic development producer. Never imported by inference.

Only this module reads oracle parameters. It generates prefix observations
before inference and held-out fields only after forecasts have been committed.
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from fatigue_dic_rl.correction.acquisition import public_patterns
from fatigue_dic_rl.correction.information_sufficiency import history_for,phase_at
from fatigue_dic_rl.correction.designed_acquisition import covariance_at
from fatigue_dic_rl.correction.law_family_benchmarks import ReferenceLawSpecification
from fatigue_dic_rl.fatigue import MaterialHistoryState,FatigueEvolutionSolver,FatigueStepRequest,FatigueSolverConfig
from fatigue_dic_rl.mechanics import MaterialAssignment
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.dic import DICObservation
from fatigue_dic_rl.observations.noise import DICMeasurement


def actual_fields(base,meta,coordinates,cycles):
    truth=json.loads((Path(base)/'synthetic_truth.json').read_text())
    original=json.loads((Path(base)/'public.json').read_text())
    reference=ReferenceLawSpecification(truth['family'],truth['A'],truth['m'],truth['strength'],(2.,1.))
    rows={k:[] for k in ('damage','u','reaction','hazard')};max_damage=0.;calls=0
    for i,p in enumerate(public_patterns(meta)):
        # True load and material are shared across counterfactual observation conditions.
        actual=replace(p,materials={r:MaterialAssignment(a.constitutive_model,dict(a.parameters,young_modulus_n_per_mm2=truth['actual_E'])) for r,a in p.materials.items()},
            traction_bcs=tuple(replace(b,values=b.values*(original['load_scales'][i]/meta['load_scales'][i])) for b in p.traction_bcs))
        layout=actual.quadrature_layout;initial=MaterialHistoryState.initialize(layout,layout.field(np.full(layout.count,truth['d0']),'1'),damage_max=.95)
        from fatigue_dic_rl.correction.robust_estimation import CountedMechanics
        mech=CountedMechanics();law=reference.build({},initial.material_ids)
        solver=FatigueEvolutionSolver(actual,initial,law,mechanics=mech,config=FatigueSolverConfig(max_delta_damage_per_step=.03))
        history=history_for(actual,meta);mapping=ObservationOperator(p.mesh).prepare(coordinates,coordinate_units='mm',coordinate_system='specimen');one={k:[] for k in rows}
        for n in cycles:
            while solver.committed_state.cycle_count<n:
                k=solver.committed_state.cycle_count;seg=next(s for s in history if s.N_start<=k<s.N_end)
                end=min(n,k+meta['step'],seg.N_end);solver.advance(FatigueStepRequest(replace(seg,N_start=k,delta_N=end-k)))
                max_damage=max(max_damage,float(solver.committed_state.damage.values.max()))
                if max_damage>=.6:raise ValueError('declared physical damage safety limit exceeded; retain failed pilot')
            s=solver.committed_state;phase=phase_at(meta['history'],n);segment=next(c for c in history if c.N_start==phase['start'])
            request=segment.request(actual,s.damage,n,'max');hi=mech.solve(request)
            from fatigue_dic_rl.fatigue import EnergyAmplitudeDriver
            lo=mech.solve(segment.request(actual,s.damage,n,'min'));driver=EnergyAmplitudeDriver().evaluate(lo,hi)
            one['damage'].append(s.damage.values);one['u'].append(mapping.apply(hi.state.displacement).values)
            one['reaction'].append(float(hi.reaction_forces.values[p.mesh.boundary_tags['left'].node_ids,0].sum()))
            one['hazard'].append(law.hazard(driver,segment,s).values)
        calls+=mech.calls
        for k in rows:rows[k].append(one[k])
    return dict(**{k:np.asarray(v) for k,v in rows.items()},cycles=np.asarray(cycles),weights=layout.physical_weights,
                actual_FEM_solves=calls,max_damage=max_damage)


def save_measurement(path,p,xy,u,cov,n,seed):
    noise=np.linalg.cholesky(cov.matrix)@np.random.default_rng(seed).normal(size=2*len(xy))
    obs=DICObservation(xy,u+noise.reshape(-1,2),np.ones(len(xy),bool),np.ones(len(xy)),n,'static',p.loading_state.load_id,p.mesh.coordinate_system,'mm','mm')
    DICMeasurement(obs,cov,np.ones(len(xy),bool)).save(path)


def produce_case(base,out,protocol,config,source_sha,config_sha):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);base=Path(base);old=json.loads((base/'public.json').read_text())
    with np.load(base/'synthetic_trajectory.npz') as z:xy=z['coordinates_mm']
    sigma=protocol.get('sigma_mm',config['study']['sigma_mm']);cycles=protocol['cycles'];cutoff=max(cycles)
    h=[]
    if protocol['loading']=='multi':
        boundaries=config['protocols']['multilevel_boundaries']
        for i,a in enumerate(config['protocols']['multilevel_amplitudes']):h.append(dict(start=boundaries[i],end=boundaries[i+1],amplitude=a,R=config['protocols']['multilevel_R'][i]))
    else:h=[dict(start=0.,end=cutoff,amplitude=1.,R=-1.)]
    future=[cutoff+x for x in config['study']['forecast_offsets']]
    h.append(dict(start=cutoff,end=max(future),amplitude=.4,R=-1.))
    options=protocol.get('options',{})
    scales=old['load_scales'] if not options.get('load_unknown') else [v/1.05 for v in old['load_scales']]
    if max(scales)*(1.05 if options.get('load_unknown') else 1.)*10>35:raise ValueError('nominal stress exceeds frozen safety envelope')
    meta=dict(schema='phase2f-development-v1',case_id=out.name,world_id=old['case_id'],seed=old['seed'],mesh=old['mesh'],material=dict(old['material'],E=2050. if options.get('material_unknown') else old['material']['E']),
        load_scales=scales,cycles=cycles,cutoff=cutoff,step=config['study']['cycle_step'],sigma_mm=sigma,noise_model=protocol.get('noise_model','independent'),history=h,options=options,healthy=protocol.get('healthy',False),reaction=protocol.get('reaction',False),future_cycles=future,units=old['units'],source_sha256=source_sha,config_sha256=config_sha)
    (out/'public.json').write_text(json.dumps(meta,indent=2)+'\n')
    fields=actual_fields(base,meta,xy,cycles);cov=covariance_at(xy,sigma,meta['noise_model']=='spatial')
    patterns=public_patterns(meta)
    for i,p in enumerate(patterns):
        for k,n in enumerate(cycles):save_measurement(out/'observations'/str(i)/str(k),p,xy,fields['u'][i,k],cov,n,old['seed']*113+i*1001+int(n))
        if meta['reaction']:
            (out/'reactions').mkdir(exist_ok=True)
            value=fields['reaction'][i,0]+np.random.default_rng(old['seed']*127+i).normal(0.,.01)
            (out/'reactions'/f'{i}.json').write_text(json.dumps(dict(value_n=value,standard_deviation_n=.01))+'\n')
        if meta['healthy']:
            from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
            params=json.loads((base/'synthetic_truth.json').read_text())
            pp=replace(p,materials={r:MaterialAssignment(a.constitutive_model,dict(a.parameters,young_modulus_n_per_mm2=params['actual_E'])) for r,a in p.materials.items()},traction_bcs=tuple(replace(b,values=b.values*(old['load_scales'][i]/scales[i])) for b in p.traction_bcs))
            healthy=Tri3MechanicsSolver().solve(pp);mapping=ObservationOperator(p.mesh).prepare(xy,coordinate_units='mm',coordinate_system='specimen')
            save_measurement(out/'healthy'/str(i),p,xy,mapping.apply(healthy.state.displacement).values,cov,0.,old['seed']*131+i)
    np.savez_compressed(out/'evaluation_prefix_truth.npz',**fields)
    (out/'producer_record.json').write_text(json.dumps(dict(base=str(base.resolve()),protocol=protocol,prefix_only=True,actual_FEM_solves=fields['actual_FEM_solves'],healthy_FEM_solves=len(patterns) if meta['healthy'] else 0,max_damage=fields['max_damage'],future_measurements_generated=False),indent=2)+'\n')
    return dict(path=str(out.resolve()),world_id=old['case_id'],case_id=out.name,protocol=protocol)
