"""Privileged synthetic generation separated from observation-only correction IO."""
from dataclasses import replace
from pathlib import Path
import json,time,hashlib
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.identifiability_verification import points
from fatigue_dic_rl.evaluation.sequential_fixtures import TruthFrame, observation_series
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.fatigue import MaterialHistoryState, FatigueEvolutionSolver, FatigueStepRequest, EnergyAmplitudeDriver
from fatigue_dic_rl.inverse.parameterizations import UniformDamage
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.registration import RegistrationError
from fatigue_dic_rl.correction.acquisition import public_patterns, public_history
from fatigue_dic_rl.correction.hybrid_damage_law import LawSpecification
from fatigue_dic_rl.correction.physics_residual import positive_hazard_log_ratio
from fatigue_dic_rl.correction.correction_basis import CorrectionBasis


def source_hash():
    files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'))+sorted(Path('configs/phase2c').glob('*'))
    return hashlib.sha256('\n'.join(str(p)+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()


def generate_correction_case(folder, *, seed, split, scenario='structural', sigma_mm=2e-7, spatial=False,
                             registration=False, load_bias=1., nx=2, ny=1, width=2., height=1., R=-1.,
                             step=2., d0=None, scales=(.15,.35,.7,1.4,2.8), beta=3., spatial_strength=0.,
                             E=2000., config_sha256='', structural_power=1, case_label=None):
    out=Path(folder);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();rng=np.random.default_rng(seed)
    d0=float(rng.uniform(.08,.16)) if d0 is None else float(d0)
    A,m=8e-5,1.
    if scenario in ('wrong_A','wrong_Am'):A*=1.6
    if scenario in ('wrong_m','wrong_Am'):m=1.18
    beta=beta if scenario in ('structural','spatial_discrepancy') else 0.
    mesh=rectangle_mesh(nx,ny,width_mm=width,height_mm=height,seed=19,jitter=.18)
    cycles=[0.,10.,20.]+[float(n) for n in range(30,221,10)]
    label = case_label or scenario
    meta={'schema':'phase2c-public-v1','case_id':f'{split}-{label}-{seed}', 'split':split,'seed':seed,
        'mesh':{'nodes':mesh.nodes.tolist(),'connectivity':[a.tolist() for a in mesh.connectivity], 'material_region_ids':mesh.material_region_ids.tolist(),
        'boundary_tags':{n:{'node_ids':v.node_ids.tolist(),'facets':v.facets.tolist()} for n,v in mesh.boundary_tags.items()},'mesh_id':mesh.mesh_id},
        'material':{'E':E,'nu':.3,'thickness':.8}, 'load_scales':list(scales),'R':R,'cycles':cycles,
        'assimilation_cutoff':20., 'future_load_factor':.25,'step':step,'nominal':{'A':8e-5,'m':1.},
        'sigma_mm':sigma_mm,'noise_model':'spatial' if spatial else 'independent','source_sha256':source_hash(),
        'config_sha256':config_sha256, 'units':{'length':'mm','stress':'N/mm^2','hazard':'cycle^-1','cycles':'cycle'},
        'shared_registration':bool(registration)}
    (out/'public.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'synthetic_truth.json').write_text(json.dumps({'d0':d0,'A':A,'m':m,'beta':beta,'spatial_strength':spatial_strength,'structural_power':structural_power,'load_bias':load_bias,'scenario':label,'discrepancy_regime':scenario},indent=2)+'\n')
    x=points(16,'random',seed=seed+101)*[width/2.,height]
    heldout=points(64,'random',seed=seed+202)*[width/2.,height]
    damage,u,node_u,stress,strain,hazard,energy,held_u=[],[],[],[],[],[],[],[]
    residual_rows=[]
    for i,p in enumerate(public_patterns(meta)):
        history=public_history(p,meta);actual=replace(p,traction_bcs=tuple(replace(bc,values=bc.values*load_bias) for bc in p.traction_bcs));layout=p.quadrature_layout
        initial=MaterialHistoryState.initialize(layout,layout.field(np.full(layout.count,d0),'1'),damage_max=.95)
        law=LawSpecification(mode='structural_reference',beta=beta,spatial_strength=spatial_strength,structural_power=structural_power).build({'A':A,'m':m},initial.material_ids)
        prior=LawSpecification().build(meta['nominal'],initial.material_ids)
        solver=FatigueEvolutionSolver(actual,initial,law)
        mapping=ObservationOperator(p.mesh).prepare(x,coordinate_units='mm',coordinate_system='specimen'); heldmap=ObservationOperator(p.mesh).prepare(heldout,coordinate_units='mm',coordinate_system='specimen')
        frames=[]; ds=[];us=[];ns=[];sts=[];ess=[];hs=[];ys=[];hdu=[]
        generated_cycles=cycles if split in ('test','ood','scientific') else cycles[:3]
        for j,target in enumerate(generated_cycles):
            while solver.committed_state.cycle_count < target:
                n=solver.committed_state.cycle_count;seg=next(c for c in history if c.N_start<=n<c.N_end);end=min(target,seg.N_end,n+step)
                row=solver.advance(FatigueStepRequest(replace(seg,N_start=n,delta_N=end-n)))
                if n < 20:
                    for sub in row.substeps:
                        true_h=law.hazard(sub.driving_used,sub.cycle,sub.state_before).values
                        prior_h=prior.hazard(sub.driving_used,sub.cycle,sub.state_before).values
                        labels,mask=positive_hazard_log_ratio(true_h,prior_h)
                        residual_rows.append({'x':sub.driving_used.energy_amplitude.values/.025,'damage':sub.state_before.damage.values,'coordinates':layout.coordinates,'target':labels,'mask':mask,'cycle':np.full(layout.count,sub.cycle.N_start),'load_index':np.full(layout.count,i),'R':np.full(layout.count,R)})
            state=solver.committed_state;seg=history[0] if target<=20 else history[1]
            cyc=replace(seg,N_start=target,delta_N=step)
            lo=solver.mechanics.solve(cyc.request(actual,state.damage,target,'min'));hi=solver.mechanics.solve(cyc.request(actual,state.damage,target,'max'))
            driving=EnergyAmplitudeDriver().evaluate(lo,hi);exact=mapping.apply(hi.state.displacement).values
            diag=replace(p,traction_bcs=tuple(replace(bc,values=bc.values*(1. if target<=20 else meta['future_load_factor'])) for bc in p.traction_bcs))
            frame=TruthFrame(target,state.damage.values,exact,hi,{'d0':d0,'A':A,'m':m});frames.append(frame)
            ds.append(state.damage.values);us.append(exact);ns.append(hi.state.displacement.values);sts.append(hi.state.stress.values);ess.append(hi.state.strain.values);hs.append(law.hazard(driving,cyc,state).values);ys.append(driving.energy_amplitude.values);hdu.append(heldmap.apply(hi.state.displacement).values)
        reg=RegistrationError([5e-5,-3e-5],4e-5) if registration else None
        observations=observation_series(p,tuple(frames),x,sigma_mm=sigma_mm,seed=seed*17+i,spatial=spatial,registration=reg,name=f'load-{i}')
        for j,f in enumerate(observations): f.case.measurement.save(out/'observations'/str(i)/str(j))
        damage.append(ds);u.append(us);node_u.append(ns);stress.append(sts);strain.append(ess);hazard.append(hs);energy.append(ys);held_u.append(hdu)
    np.savez_compressed(out/'synthetic_trajectory.npz',cycles=np.asarray(generated_cycles),damage=np.asarray(damage),displacement_mm=np.asarray(u),node_displacement_mm=np.asarray(node_u),stress=np.asarray(stress),strain=np.asarray(strain),hazard=np.asarray(hazard),energy=np.asarray(energy),heldout_displacement_mm=np.asarray(held_u),heldout_coordinates_mm=heldout,coordinates_mm=x,quadrature_coordinates_mm=layout.coordinates,physical_weights_mm2=layout.physical_weights)
    keys=residual_rows[0].keys()
    np.savez_compressed(out/'privileged_prefix_hazard.npz',**{k:np.concatenate([row[k] for row in residual_rows]) for k in keys})
    return {'path':str(out.resolve()),'case_id':meta['case_id'],'seed':seed,'split':split,'scenario':label,'wall_seconds':time.perf_counter()-t,'public_sha256':hashlib.sha256((out/'public.json').read_bytes()).hexdigest(),'truth_sha256':hashlib.sha256((out/'synthetic_trajectory.npz').read_bytes()).hexdigest()}


def load_privileged_prefix(folder, *, allowed_splits=('train','validation'), dimension=2, basis_kind='energy'):
    p=Path(folder);meta=json.loads((p/'public.json').read_text())
    if meta['split'] not in allowed_splits:raise ContractError('training cannot read calibration/test/OOD hazard labels')
    with np.load(p/'privileged_prefix_hazard.npz',allow_pickle=False) as a:
        if np.any(a['cycle']>=20):raise ContractError('hazard training labels exceed the observed prefix')
        mask=a['mask']; x=a['x'][mask]; d=a['damage'][mask];coords=a['coordinates'][mask];targets=a['target'][mask]
    if not np.isfinite(targets).all():raise ContractError('invalid privileged label')
    if not len(targets):raise ContractError('no positive-hazard supervision available; zero-hazard points cannot create ratio labels')
    return CorrectionBasis(dimension,basis_kind).evaluate(x,d,coords),targets
