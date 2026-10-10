"""Small CPU/GPU reproducible supervised training; SciPy features are detached.

Validation selects epochs; calibration/test labels are never opened here.
The observation-only control uses TRF pseudo-labels from train acquisitions.
"""
from dataclasses import replace
import hashlib,json
from pathlib import Path
import time
import numpy as np
import torch
from fatigue_dic_rl.networks.temporal_gru import NetworkConfig,GraphRecurrentEstimator,pack_graph_frames
from .synthetic_dataset import load_graph_cache,load_parameter_labels,load_observations,source_hash
from .losses import parameter_loss,gaussian_parameter_nll
from .checkpoint import save_checkpoint


def forward_model(model,sequence,settings):
    if settings.get('causal_uniform_physics',False) and model.config.physics:
        from .uniform_condition import forward_causal
        return forward_causal(model,sequence)
    return model(sequence)


def observation_only_targets(record):
    from fatigue_dic_rl.inverse.parameterizations import UniformDamage,ParameterSpace
    from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
    from fatigue_dic_rl.inverse.objectives import InverseObjective
    from fatigue_dic_rl.inverse.optimizer import invert
    _,predictor,frames=load_observations(record['path'],cutoff=0.)
    model=UniformDamage();adapter=StaticForwardAdapter(model,ParameterSpace(model.parameters),[frames[0].case])
    result=invert(InverseObjective(adapter),[np.zeros(1)]).best
    if result is None or not result.converged:raise RuntimeError('observation-only pseudo-label optimizer failed')
    d=adapter.parameter_space.decode(result.normalized)['d0']
    return predictor.parameter_space.encode({'d0':d,'A':.0015})


def read_partition(manifest,split,*,observation_only=False,causal_physics=False):
    if split not in ('train','validation'):raise ValueError('training loader refuses calibration/test partitions')
    rows=[]
    for r in manifest['records']:
        if r['split']!=split:continue
        views=load_graph_cache(Path(r['path'])/'training_condition_features.npz')
        if causal_physics:
            from fatigue_dic_rl.networks.observation_graph import equivariant_whitening
            _,_,frames=load_observations(r['path']);view=[]
            for graph,frame in zip(views[0],frames):
                m=frame.case.measurement;mask=m.active_mask;cov=m.covariance.selected_matrix(mask)
                basis=np.column_stack([graph.dic.coordinates_mm[:,0],-.3*graph.dic.coordinates_mm[:,1]])
                measured=equivariant_whitening(cov,m.observation.displacement[mask],frame.case.variance_floor_mm2)[0]
                white_basis=equivariant_whitening(cov,basis,frame.case.variance_floor_mm2)[0]
                view.append(replace(graph,training_whitened_measured=measured,training_whitened_affine_basis=white_basis))
            views=[view]  # own causal prediction replaces all teacher conditions
        # Control never opens internal labels, including validation labels.
        target=observation_only_targets(r) if observation_only else load_parameter_labels(r['path'])['normalized_parameters']
        rows.append((views,np.asarray(target,dtype=np.float32)))
    return rows


def mini_batches(rows,indices,rng,*,batch_size,device,physics,recurrent,training=False):
    for begin in range(0,len(indices),batch_size):
        selected=[rows[i] for i in indices[begin:begin+batch_size]];graphs=[];targets=[]
        for views,target in selected:
            vi=int(rng.integers(len(views))) if training else 0;graphs.append(views[vi] if recurrent else [views[vi][0]]);targets.append(target)
        sequence=[pack_graph_frames([g[k] for g in graphs],device=device,physics=physics) for k in range(len(graphs[0]))]
        yield sequence,torch.as_tensor(np.array(targets),device=device)


def point_losses(model,outputs,target,sequence):
    losses=[]
    for output,data in zip(outputs,sequence):
        desired=target
        if model.config.recurrent and model.config.parameter_names==('d0','A'):
            # A has exactly zero sensitivity at N=0. Its conditional point center
            # is the declared training-prior mean; covariance learns true errors.
            desired=target.clone();at_zero=data['global'][:,18]==0.
            desired[at_zero,1]=(.0015-.00205)/.00195
        losses.append(parameter_loss(output['normalized_mean'],desired))
    return losses


def validation_loss(model,rows,settings,device):
    model.eval();s=0.;n=0
    with torch.no_grad():
        for sequence,target in mini_batches(rows,np.arange(len(rows)),np.random.default_rng(0),batch_size=settings['batch_size'],device=device,physics=model.config.physics,recurrent=model.config.recurrent):
            outputs=forward_model(model,sequence,settings);loss=torch.stack(point_losses(model,outputs,target,sequence)).mean();s+=float(loss)*len(target);n+=len(target)
    return s/n


def train_one(config,manifest,output,variant,seed,*,device='cpu',initial_model=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);settings=config['training'];torch.set_num_threads(settings.get('threads',1));torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed);begin=time.perf_counter();observation_only=variant=='graph_observation_only';cov_only=variant=='physics_shape_uncertainty'
    causal=settings.get('causal_uniform_physics',False) and variant.startswith('physics')
    train=read_partition(manifest,'train',observation_only=observation_only,causal_physics=causal);validation=read_partition(manifest,'validation',observation_only=observation_only,causal_physics=causal)
    cfg=NetworkConfig(parameter_names=tuple(config['network']['parameter_names']),hidden_channels=config['network']['hidden_channels'],graph_layers=config['network']['graph_layers'],recurrent=variant not in ('graph_initial','graph_observation_only'),physics=variant.startswith('physics'),fusion='global' if variant=='physics_global' else 'shape',uncertainty=cov_only)
    model=GraphRecurrentEstimator(cfg).to(device)
    if initial_model is not None:model.load_state_dict(initial_model.state_dict())
    if cov_only:
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('uncertainty_head.'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'],weight_decay=settings['weight_decay']);history=[];best=float('inf');best_state=None;stale=0;epochs=settings['covariance_epochs'] if cov_only else settings['epochs']
    if initial_model is not None and not cov_only:
        best=validation_loss(model,validation,settings,device);best_epoch=-1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for epoch in range(epochs):
        model.train();loss_sum=0.;n=0;ep_begin=time.perf_counter()
        for sequence,target in mini_batches(train,rng.permutation(len(train)),rng,batch_size=settings['batch_size'],device=device,physics=cfg.physics,recurrent=cfg.recurrent,training=True):
            optimizer.zero_grad(set_to_none=True);outputs=forward_model(model,sequence,settings)
            losses=[gaussian_parameter_nll(o['normalized_mean'].detach(),o['cholesky'],target) for o in outputs] if cov_only else point_losses(model,outputs,target,sequence)
            loss=torch.stack(losses).mean();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],settings['gradient_clip']);optimizer.step();loss_sum+=float(loss.detach())*len(target);n+=len(target)
        if cov_only:
            model.eval();val_sum=vn=0
            with torch.no_grad():
                for sequence,target in mini_batches(validation,np.arange(len(validation)),rng,batch_size=settings['batch_size'],device=device,physics=cfg.physics,recurrent=cfg.recurrent):
                    outputs=forward_model(model,sequence,settings);v=torch.stack([gaussian_parameter_nll(o['normalized_mean'],o['cholesky'],target) for o in outputs]).mean();val_sum+=float(v)*len(target);vn+=len(target)
            val=val_sum/vn
        else:val=validation_loss(model,validation,settings,device)
        history.append({'epoch':epoch,'train_loss':loss_sum/n,'validation_loss':val,'epoch_seconds':time.perf_counter()-ep_begin})
        if val<best:
            best=val;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        minimum=settings.get('minimum_recurrent_point_epochs',0) if cfg.recurrent and not cov_only else 0
        if stale>=settings['patience'] and epoch+1>=minimum:break
    model.load_state_dict(best_state);model.eval()
    metadata={'variant':variant,'seed':seed,'device':device,'synthetic_supervised':not observation_only,'observation_only_control':'TRF initial-damage pseudo-label from train/validation observed N0; A fixed declared prior, not truth' if observation_only else None,'source_sha256':config.get('_training_source_sha256',source_hash()),'dataset_source_sha256':manifest['source_sha256'],'dataset_config_sha256':manifest['config_sha256'],'training_config':settings,'best_epoch':best_epoch,'validation_loss':best,'epochs_executed':len(history),'wall_seconds':time.perf_counter()-begin,'torch_version':str(torch.__version__),'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,'differentiation':'torch graph/GRU/head only; cached SciPy physics is stopped gradient','teacher_forcing':'train-only mixed nominal and perturbed-label physics conditions; validation conditions nominal; no-physics GRU prior feature removed','zero_cycle_fatigue_point_center':'declared training-prior mean for recurrent point head; uncertainty trained with actual A labels','calibration_test_access':False}
    if causal:
        metadata['teacher_forcing']='none: each step uses own previous mean; stopped-gradient verified uniform equivalent conditions'
        metadata['differentiation']='torch graph/GRU/head only; scalar homogeneous equivalent physics conditions detached; production SciPy inference'
    digest=save_checkpoint(out/'checkpoint.pt',model,metadata);metadata['checkpoint_sha256']=digest
    (out/'history.json').write_text(json.dumps(history,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n');print(variant,seed,'epochs',len(history),'best',best,'wall',metadata['wall_seconds'],flush=True);return model,metadata
