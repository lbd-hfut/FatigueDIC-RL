"""Virtual FE-damage experiments conditioned only on a measured N0/10 prefix."""
from dataclasses import asdict
import numpy as np
from fatigue_dic_rl.inverse.sensitivity import centered_jacobian
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter
from fatigue_dic_rl.correction.identifiability_gate import conditional_singular_values
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference
from fatigue_dic_rl.correction.designed_acquisition import predictors_for,public_designs,blank_batches,covariance_at


def point_pool(prefix,seed):
    x=prefix[0].frames[0].case.measurement.observation.coordinates;axisx=np.linspace(.06,1.94,8);axisy=np.linspace(.04,.96,8);grid=np.array([(a,b) for a in axisx for b in axisy]);pool=np.vstack([grid,x])
    uniform=np.array([8*i+j for i in (0,2,5,7) for j in (0,2,5,7)]);outer=np.array([8*i+j for i in (0,7) for j in range(8)]);random=np.random.default_rng(seed+1001).choice(64,16,replace=False)
    return pool,dict(original=np.arange(64,80),uniform=uniform,outer=outer,random=random)


def virtual_library(patterns,prefix,meta,options,fits,*,amplitudes=(.25,.4,.7,1.),ratios=(-1.,0.),future_R=-1.,damage_limit=.6,stress_limit=35.):
    pool,sets=point_pool(prefix,meta['seed']);cov=covariance_at(pool,meta['sigma_mm'],meta['noise_model']=='spatial');library=[];virtual_solves=0;fields={}
    for amp in amplitudes:
        for R in ratios:
            model={};failed=None
            for method in ('B2','B5'):
                ps=None
                try:
                    ps=predictors_for(patterns,method,meta,options,amplitude=amp,R=R,future_R=future_R);designs=public_designs(patterns,pool,meta,options,amplitude=amp);adapter=MultiTrajectoryAdapter(ps,blank_batches(designs,options));z=ps[0].parameter_space.encode(fits[method].physical);y=adapter.displacement_prediction(z);s,derivative_method=checked_local_jacobian(adapter.displacement_prediction,z)
                    max_damage=max(float(p.damage_states(z,(20.,))[20.].damage.values.max()) for p in ps)
                    model[method]=dict(y=y.reshape(5,80,2),j=s.jacobian.reshape(5,80,2,-1),sensitivity=s,space=ps[0].parameter_space,max_damage=max_damage,derivative_method=derivative_method)
                except Exception as exc:failed=type(exc).__name__+': '+str(exc);break
                finally:
                    if ps is not None:virtual_solves+=sum(p.mechanics.calls for p in ps)
            for name,indices in sets.items():
                key=f'a{amp:g}_R{R:g}_{name}'
                if failed is not None:library.append(dict(design_id=key,amplitude=amp,R=R,point_set=name,indices=indices,safe=False,derivative_checked=False,fisher_score=0.,projected_separation=0.,failure=failed));continue
                mask=np.zeros(80,bool);mask[indices]=True;factor=cov.factor(mask);# factor order follows sorted point indices
                idx=np.flatnonzero(mask);white={};info={};nuisances={};minimums=[];parameter_minima=[]
                for method,m in model.items():
                    white[method]=np.concatenate([factor.whiten(y[idx].ravel()) for y in m['y']]);j=np.concatenate([factor.whiten(j[idx].reshape(2*len(idx),-1)) for j in m['j']],axis=0);info[method]=j
                    names=m['space'].names;target=[i for i,n in enumerate(names) if n in ('A','m','c0','c1')];combined=np.vstack([fits[method].uncertainty.jacobian_whitened,j]);sv=conditional_singular_values(combined,target);minimums.append(float(sv[-1]) if len(sv) else 0.)
                    parameter_minima.append(float(np.linalg.svd(combined,compute_uv=False)[-1]))
                    nuisance=[i for i,n in enumerate(names) if n in ('E0','load_factor','tx','ty','theta')];nuisances[method]=j[:,nuisance]
                difference=white['B2']-white['B5'];projected=[project_whitened_difference(difference,nuisances[m],numerical_resolution=model[m]['sensitivity'].absolute_error_norm*float(np.linalg.norm(factor.whiten(np.eye(2*len(idx))),2))*np.sqrt(5)) for m in ('B2','B5')];separation=min(np.sqrt(p.projected_squared_distance) for p in projected);fisher=min(minimums)**2
                nominal_stress=max(meta['load_scales'])*10*amp*max(fits[m].physical.get('load_factor',1.) for m in model)
                safe=all(m['max_damage']<damage_limit for m in model.values()) and nominal_stress<=stress_limit;checked=all(m['sensitivity'].converged for m in model.values())
                library.append(dict(design_id=key,amplitude=amp,R=R,point_set=name,indices=idx,safe=safe,derivative_checked=checked,fisher_score=fisher,projected_separation=float(separation),raw_separation=float(np.linalg.norm(difference)),evolution_singular_minima=minimums,parameter_singular_minima=parameter_minima,maximum_predicted_damage=max(m['max_damage'] for m in model.values()),measurement_cost=160,fatigue_cycles_per_specimen=10.))
                fields[key+'_difference']=difference;fields[key+'_projected_B2']=projected[0].projected_difference;fields[key+'_projected_B5']=projected[1].projected_difference
    return library,pool,fields,virtual_solves
