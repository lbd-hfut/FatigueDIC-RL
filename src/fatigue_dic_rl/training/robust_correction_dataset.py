"""Independent reference-law rollout; truth storage is separate from acquisition."""
from pathlib import Path
from dataclasses import replace
import json, hashlib, time
import numpy as np
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.identifiability_verification import points
from fatigue_dic_rl.evaluation.sequential_fixtures import TruthFrame, observation_series
from fatigue_dic_rl.correction.acquisition import public_patterns, public_history
from fatigue_dic_rl.correction.law_family_benchmarks import ReferenceLawSpecification
from fatigue_dic_rl.fatigue import MaterialHistoryState, FatigueEvolutionSolver, FatigueStepRequest, EnergyAmplitudeDriver
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.registration import RegistrationError


def source_hash():
    paths=sorted(p for folder in ('src','scripts','configs/phase2d') for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ('.py','.toml','.json') and '__pycache__' not in p.parts)
    return hashlib.sha256('\n'.join(str(p)+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()).hexdigest()


def generate_case(folder, spec, *, config_sha256):
    out=Path(folder);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();rng=np.random.default_rng(spec['seed'])
    d0=float(rng.uniform(.08,.16));E=float(rng.uniform(1800,2200));loadscale=float(rng.uniform(.92,1.08));future_factor=.25 if spec['seed']%2 else .4
    A,m=8e-5,1.
    if spec['family']=='parameter': A*=float(rng.uniform(1.35,1.65));m=float(rng.uniform(1.12,1.22))
    family=spec['family'];strength={'matched':0.,'parameter':0.,'S1':3.,'S2':.5,'S3':.55,'S4':3.,'S5':.3}[family]
    nx,ny=spec.get('nx',2),spec.get('ny',1);w,h=spec.get('width',2.),spec.get('height',1.);R=spec.get('R',-1.)
    mesh=rectangle_mesh(nx,ny,width_mm=w,height_mm=h,seed=19,jitter=.18);cycles=[0.,10.,20.,30.,40.,70.,120.,220.]
    meta=dict(schema='phase2c-public-v1',case_id=spec['case_id'],split=spec['split'],seed=spec['seed'],mesh=dict(nodes=mesh.nodes.tolist(),connectivity=[a.tolist() for a in mesh.connectivity],material_region_ids=mesh.material_region_ids.tolist(),boundary_tags={n:dict(node_ids=v.node_ids.tolist(),facets=v.facets.tolist()) for n,v in mesh.boundary_tags.items()},mesh_id=mesh.mesh_id),material=dict(E=E,nu=.3,thickness=.8),load_scales=(loadscale*np.array([.15,.35,.7,1.4,2.8])).tolist(),R=R,cycles=cycles,assimilation_cutoff=20.,future_load_factor=future_factor,step=spec.get('step',2.),nominal=dict(A=8e-5,m=1.),sigma_mm=spec['sigma_mm'],noise_model='spatial' if spec.get('spatial') else 'independent',source_sha256=source_hash(),config_sha256=config_sha256,units=dict(length='mm',stress='N/mm^2',hazard='cycle^-1',cycles='cycle'),shared_registration=bool(spec.get('registration')))
    (out/'public.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'generation_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    q=spec.get('Q',16);xy=points(q,'random',seed=spec['seed']+101)*[w/2.,h]
    if spec.get('clustered'): xy=np.random.default_rng(spec['seed']+101).uniform([.75*w,.40*h],[.95*w,.55*h],(q,2))
    held=points(64,'random',seed=spec['seed']+202)*[w/2.,h]
    reference=ReferenceLawSpecification(family,A,m,strength,(w,h));load_bias=spec.get('load_bias',1.);actual_E=E*spec.get('E_bias',1.)
    arrays={n:[] for n in ('damage','displacement_mm','node_displacement_mm','stress','strain','hazard','energy','heldout_displacement_mm')};reactions=[]
    from fatigue_dic_rl.mechanics import MaterialAssignment
    generated=cycles if spec['split'] in ('test','ood','diagnostic') else cycles[:3]
    for i,p in enumerate(public_patterns(meta)):
        actual=replace(p,materials={r:MaterialAssignment(a.constitutive_model,dict(a.parameters,young_modulus_n_per_mm2=actual_E)) for r,a in p.materials.items()},traction_bcs=tuple(replace(b,values=b.values*load_bias) for b in p.traction_bcs))
        layout=p.quadrature_layout;state=MaterialHistoryState.initialize(layout,layout.field(np.full(layout.count,d0),'1'),damage_max=.95);law=reference.build({},state.material_ids);solver=FatigueEvolutionSolver(actual,state,law);history=public_history(p,meta)
        mapping=ObservationOperator(p.mesh).prepare(xy,coordinate_units='mm',coordinate_system='specimen');hmap=ObservationOperator(p.mesh).prepare(held,coordinate_units='mm',coordinate_system='specimen');rows={n:[] for n in arrays};frames=[]
        for n in generated:
            while solver.committed_state.cycle_count<n:
                k=solver.committed_state.cycle_count;seg=next(s for s in history if s.N_start<=k<s.N_end);end=min(n,seg.N_end,k+meta['step']);solver.advance(FatigueStepRequest(replace(seg,N_start=k,delta_N=end-k)))
            state=solver.committed_state;seg=history[0] if n<=20 else history[1];cyc=replace(seg,N_start=n,delta_N=meta['step']);lo=solver.mechanics.solve(cyc.request(actual,state.damage,n,'min'));hi=solver.mechanics.solve(cyc.request(actual,state.damage,n,'max'));driving=EnergyAmplitudeDriver().evaluate(lo,hi);u=mapping.apply(hi.state.displacement).values
            frames.append(TruthFrame(n,state.damage.values,u,hi,dict(d0=d0,A=A,m=m)))
            vals=dict(damage=state.damage.values,displacement_mm=u,node_displacement_mm=hi.state.displacement.values,stress=hi.state.stress.values,strain=hi.state.strain.values,hazard=law.hazard(driving,cyc,state).values,energy=driving.energy_amplitude.values,heldout_displacement_mm=hmap.apply(hi.state.displacement).values)
            for name in arrays:rows[name].append(vals[name])
        reg=RegistrationError([5e-5,-3e-5],4e-5) if spec.get('registration') else None
        observations=observation_series(p,tuple(frames),xy,sigma_mm=spec['sigma_mm'],seed=spec['seed']*17+i,spatial=spec.get('spatial',False),registration=reg,missing=spec.get('missing',0.))
        for j,f in enumerate(observations):f.case.measurement.save(out/'observations'/str(i)/str(j))
        reaction=-10*meta['load_scales'][i]*.8*h*load_bias+np.random.default_rng(spec['seed']+i+777).normal(0.,.01);reactions.append(dict(value_n=reaction,std_n=.01))
        for name in arrays:arrays[name].append(rows[name])
    np.savez_compressed(out/'synthetic_trajectory.npz',cycles=generated,**{k:np.asarray(v) for k,v in arrays.items()},coordinates_mm=xy,heldout_coordinates_mm=held,quadrature_coordinates_mm=layout.coordinates,physical_weights_mm2=layout.physical_weights)
    (out/'synthetic_truth.json').write_text(json.dumps(dict(d0=d0,A=A,m=m,family=family,strength=strength,actual_E=actual_E,load_bias=load_bias),indent=2)+'\n');(out/'observed_reaction.json').write_text(json.dumps(reactions,indent=2)+'\n')
    return dict(**spec,path=str(out.resolve()),wall_seconds=time.perf_counter()-start,public_sha256=hashlib.sha256((out/'public.json').read_bytes()).hexdigest(),truth_sha256=hashlib.sha256((out/'synthetic_trajectory.npz').read_bytes()).hexdigest())
