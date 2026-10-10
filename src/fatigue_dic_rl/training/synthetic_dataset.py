"""Deterministic case-level splits; truth is separated from public observations.

Every trajectory comes from the verified Tri3 + Phase 1B fatigue model.
Only train caches may condition teacher-forcing physics on synthetic labels.
Validation caches use the declared prior, never true initialization.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib,json
from pathlib import Path
import time
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.evaluation.identifiability_verification import pattern_for,material,points
from fatigue_dic_rl.evaluation.sequential_fixtures import generate_truth,observation_series,fatigue_loading
from fatigue_dic_rl.inverse.parameterizations import UniformDamage,GaussianDamage,Parameter,ParameterSpace
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase
from fatigue_dic_rl.inference.belief_state import ObservationFrame
from fatigue_dic_rl.inference.physics_predictor import PhysicsPredictor
from fatigue_dic_rl.observations.noise import DICMeasurement
from fatigue_dic_rl.observations.registration import RegistrationError
from fatigue_dic_rl.networks.graph_features import build_graph_frame,GraphFrame
from fatigue_dic_rl.networks.observation_graph import GraphArrays


def source_hash():
    files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'))
    return hashlib.sha256('\n'.join(str(p)+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()


def public_model(metadata):
    g=metadata['geometry'];mesh=rectangle_mesh(g['nx'],g['ny'],width_mm=g['width_mm'],height_mm=g['height_mm'],seed=g['seed'],jitter=.18)
    pattern=replace(pattern_for(mesh),materials=material(metadata['young_modulus_n_mm2'],.3))
    model=GaussianDamage() if metadata.get('damage_model')=='gaussian' else UniformDamage()
    specs=model.parameters+((Parameter('A',.0001,.004,'cycle^-1'),) if isinstance(model,UniformDamage) else ())
    space=ParameterSpace(specs,{'A':metadata.get('known_fatigue_A',.0015)} if isinstance(model,GaussianDamage) else {})
    load=fatigue_loading(pattern)
    predictor=PhysicsPredictor(model,space,pattern,load_history=load)
    return model,space,pattern,predictor


def prior_normalized(space):
    if space.names==('d0','A'):return space.encode({'d0':.16,'A':.0015})
    return np.zeros(len(space.names))


def load_observations(case_folder, *, cutoff=20.):
    """Choose public schedule BEFORE opening files; never opens labels/future."""
    p=Path(case_folder);meta=json.loads((p/'public.json').read_text());model,space,pattern,predictor=public_model(meta)
    frames=[]
    for i,n in enumerate(meta['cycles']):
        if n>cutoff:continue
        measurement=DICMeasurement.load(p/'observations'/str(i))
        frames.append(ObservationFrame(f"{meta['case_id']}-{i}",StaticLoadCase(pattern,measurement,None,1e-10 if meta['sigma_mm']==0 else 0.)))
    return meta,predictor,tuple(frames)


def load_labels(case_folder):
    with np.load(Path(case_folder)/'synthetic_labels.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def load_parameter_labels(case_folder):
    """Supervision/calibration only; never materializes future damage/u arrays."""
    with np.load(Path(case_folder)/'synthetic_labels.npz',allow_pickle=False) as a:
        return {k:a[k].copy() for k in ('normalized_parameters','physical_parameters')}


def save_graph_cache(path, views):
    data={}
    for vi,view in enumerate(views):
        for k,g in enumerate(view):
            prefix=f'v{vi}_t{k}_'
            for n,x in {'dx':g.dic.coordinates_mm,'df':g.dic.features,'de':g.dic.edge_index,'fx':g.fe.coordinates_mm,'ff':g.fe.features,'fe':g.fe.edge_index,'ids':g.mapping_node_ids,'weights':g.mapping_weights,'global':g.global_features,'previous':g.previous_normalized}.items():data[prefix+n]=x
    data['view_count']=np.array(len(views));data['frame_count']=np.array(len(views[0]));np.savez_compressed(path,**data)


def load_graph_cache(path):
    views=[]
    with np.load(path,allow_pickle=False) as a:
        for vi in range(int(a['view_count'])):
            view=[]
            for k in range(int(a['frame_count'])):
                prefix=f'v{vi}_t{k}_';get=lambda n:a[prefix+n].copy()
                view.append(GraphFrame(GraphArrays(get('dx'),get('df'),get('de')),GraphArrays(get('fx'),get('ff'),get('fe')),get('ids'),get('weights'),get('global'),get('previous')))
            views.append(view)
    return views


def generate_case(task):
    config,split,seed,folder,scenario=task;out=Path(folder);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();rng=np.random.default_rng(seed);d=config['dataset']
    nx=int(rng.choice(d['mesh_nx']));q=int(rng.choice(d['observation_counts']));sampling='regular' if rng.random()<.2 else 'random'
    g={'nx':nx,'ny':max(1,nx//2),'width_mm':float(rng.uniform(1.8,2.2)),'height_mm':float(rng.uniform(.9,1.1)),'seed':seed}
    young=float(rng.uniform(1800.,2200.));sigma=float(rng.uniform(*d['sigma_range_mm']));missing=.1 if rng.random()<.3 else 0.;spatial=rng.random()<.2
    overrides={'q16':16,'q64':64,'q256':256,'q37':37}
    if scenario in overrides:q=overrides[scenario];sampling='random';spatial=False
    if scenario in ('clustered','regular'):sampling=scenario
    if scenario=='cross_mesh':g['nx']=8;g['ny']=4
    if scenario=='cross_geometry':g['width_mm']=3.;g['height_mm']=.7
    if scenario=='heldout_material':young=2600.
    if scenario=='spatial':spatial=True
    if scenario=='independent':spatial=False
    if scenario=='no_noise':sigma=0.;spatial=False;missing=0.
    if scenario=='missing':missing=.45
    metadata={'case_id':f'{split}-{scenario}-{seed}','split':split,'seed':seed,'scenario':scenario,'geometry':g,'young_modulus_n_mm2':young,'sigma_mm':sigma,'cycles':d['cycles'],'assimilation':d['assimilation'],'forecast':d['forecast'],'damage_model':'uniform','units':{'length':'mm','displacement':'mm','E':'N/mm^2','A':'cycle^-1'},'noise_contract':'independent_given_shared_parameters','source_sha256':config.get('_generation_source_sha256',source_hash()),'synthetic_supervised_labels':True,'public_load_max':1.,'public_load_min':-1.}
    if scenario=='irregular_time':metadata['cycles']=[0.,5.,20.,30.,40.];metadata['assimilation']=[0.,5.,20.]
    if config.get('case')=='gaussian_known_fatigue':metadata.update(damage_model='gaussian',known_fatigue_A=.0015)
    model,space,pattern,predictor=public_model(metadata)
    if isinstance(model,UniformDamage):physical={'d0':float(rng.uniform(*d['d0_range'])),'A':float(rng.uniform(*d['fatigue_A_range']))}
    else:
        ranges=[(.02,.08),(.15,.4),(.7,1.3),(.3,.7),(.25,.5),(.15,.35)];physical={name:float(rng.uniform(*bounds)) for name,bounds in zip(space.names,ranges)};physical['A']=.0015
    actual=dict(physical);registration=None
    if scenario=='registration':
        registration=RegistrationError([.0005,-.0003],.0004);actual.update(tx=.0005,ty=-.0003,theta=.0004)
    if scenario=='unknown_loading':actual['load_factor']=1.1
    if scenario=='wrong_fixed_A':actual['A']=min(.003,actual['A']*1.7)
    exponent=2. if scenario=='model_discrepancy' else 1.
    x=points(q,sampling,seed=seed+67)*np.array([g['width_mm']/2.,g['height_mm']])
    if scenario=='irregular_time':
        truths=tuple(generate_truth(pattern,model,actual,[n],x,load_history=fatigue_loading(pattern),law_exponent=exponent)[0] for n in metadata['cycles'])
    else:truths=generate_truth(pattern,model,actual,metadata['cycles'],x,load_history=fatigue_loading(pattern),law_exponent=exponent)
    frames=observation_series(pattern,truths,x,sigma_mm=sigma,seed=seed+131,spatial=spatial,registration=registration,missing=missing)
    for i,f in enumerate(frames):f.case.measurement.save(out/'observations'/str(i))
    metadata.update(noise_model='spatial_Gaussian' if spatial else 'independent_Gaussian',missing_probability=missing,sampling=sampling)
    (out/'public.json').write_text(json.dumps(metadata,indent=2)+'\n')
    np.savez_compressed(out/'synthetic_labels.npz',parameter_names=np.array(space.names),normalized_parameters=space.encode(actual),physical_parameters=np.array([actual[n] for n in space.names]),damage=np.array([t.damage for t in truths]),exact_displacement_mm=np.array([t.displacement_dic_mm for t in truths]),node_displacement_mm=np.array([t.response.state.displacement.values for t in truths]),cycles=np.array(metadata['cycles']),coordinates_mm=x)
    # No forecast measurement or truth is used to form a graph cache.
    if split in ('train','validation'):
        prefix=[f for f in frames if f.cycle_count<=20.];nominal=prior_normalized(space);views=[]
        for vi in range(2 if split=='train' else 1):
            view=[]
            for k,f in enumerate(prefix):
                previous=nominal.copy()
                if vi==1 and k>0:
                    # Synthetic-supervised teacher forcing only; independent bounded perturbations.
                    previous=np.clip(space.encode(physical)+rng.normal(0.,[.12,.20] if len(space.names)==2 else np.full(6,.15)),-.95,.95)
                view.append(build_graph_frame(f,previous,predictor=predictor,physics=True,previous_cycle=prefix[k-1].cycle_count if k else 0.))
            views.append(view)
        save_graph_cache(out/'training_condition_features.npz',views)
    return {'case_id':metadata['case_id'],'split':split,'seed':seed,'scenario':scenario,'path':str(out),'public_sha256':hashlib.sha256((out/'public.json').read_bytes()).hexdigest(),'label_sha256':hashlib.sha256((out/'synthetic_labels.npz').read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-t,'parameter_names':space.names,'truth_law_exponent':exponent}


def generate_dataset(config, output, *, config_bytes=b''):
    config=dict(config,_generation_source_sha256=source_hash())
    out=Path(output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();d=config['dataset'];tasks=[];seed=d['first_seed']
    for split in ('train','validation','calibration','test'):
        for i in range(d[split]):
            tasks.append((config,split,seed,str(out/split/f'case_{i:04d}'),'in_distribution'));seed+=1
    with ProcessPoolExecutor(max_workers=d.get('workers',1)) as pool:records=list(pool.map(generate_case,tasks))
    seeds=[r['seed'] for r in records]
    if len(set(seeds))!=len(seeds):raise ContractError('trajectory seeds overlap across splits')
    manifest={'records':records,'source_sha256':config['_generation_source_sha256'],'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'split_rule':'distinct base physics/trajectory seeds, never split frames/noise variants across partitions','truth_supervision':'synthetic supervised only; observation-only inference does not load labels','teacher_forcing':'train only; validation conditions use declared nominal, not truth','wall_seconds':time.perf_counter()-t}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
