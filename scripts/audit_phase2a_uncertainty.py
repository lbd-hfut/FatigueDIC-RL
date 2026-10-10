"""Independent, predeclared follow-up; never writes Phase 2A assets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time
import tomllib
import numpy as np
from scipy.optimize import least_squares
from fatigue_dic_rl.evaluation.sequential_verification import _calibration_trial, aggregate_calibration
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup, generate_truth, observation_series, fatigue_loading
from fatigue_dic_rl.evaluation.identifiability_verification import clean
from fatigue_dic_rl.inverse.parameterizations import Parameter
from fatigue_dic_rl.inference.physics_predictor import PhysicsForwardAdapter
from fatigue_dic_rl.inference.belief_state import ObservationBatch
from fatigue_dic_rl.observations.registration import RegistrationError


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/phase2b/uncertainty_followup.toml'); ap.add_argument('--output', required=True)
    args = ap.parse_args(); c = tomllib.loads(Path(args.config).read_text()); out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    (out/'config.toml').write_bytes(Path(args.config).read_bytes()); t = time.perf_counter()
    seeds = list(range(c['calibration']['first_seed'], c['calibration']['first_seed']+c['calibration']['seeds']))
    with ProcessPoolExecutor(max_workers=c['calibration']['workers']) as pool:
        records = list(pool.map(_calibration_trial, [(c, True, 'registration', s, str(out)) for s in seeds]))
    summary = aggregate_calibration(records, seeds)
    seed = c['calibration']['profile_seed']; raw = np.load(out/f'registration_seed_{seed}/fit_prefix_0.npz')
    extra = (Parameter('tx', -.002, .002, 'mm'), Parameter('ty', -.002, .002, 'mm'), Parameter('theta', -.003, .003, 'rad'))
    model, space, pattern, predictor, x = uniform_setup(c, fatigue=True, extra_parameters=extra)
    physical = dict(d0=.12, A=.0015, tx=.0005, ty=-.0003, theta=.0004)
    truths = generate_truth(pattern, model, physical, [0.,10.,20.], x, load_history=fatigue_loading(pattern))
    frames = observation_series(pattern, truths, x, sigma_mm=c['calibration']['sigma_mm'], seed=seed,
                                registration=RegistrationError([physical['tx'],physical['ty']],physical['theta']))
    adapter = PhysicsForwardAdapter(predictor, ObservationBatch(frames, 'independent_given_shared_registration'))
    fit = least_squares(adapter.residual, np.zeros(5), bounds=(-1.,1.), method='trf', jac='3-point', diff_step=1e-4, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    z = fit.x; F = fit.jac.T@fit.jac; C = np.linalg.inv(F); sd = np.sqrt(np.diag(C)); corr = C/np.sqrt(np.outer(np.diag(C),np.diag(C)))
    objective = lambda zz: float(.5*np.sum(adapter.residual(zz)**2))
    # Full observed Hessian by centered differences of the scalar NLL; includes residual curvature.
    h = 2e-4; H = np.zeros((5,5)); f0=objective(z)
    for i in range(5):
        ei=np.eye(5)[i]*h
        H[i,i]=(objective(z+ei)-2*f0+objective(z-ei))/h**2
        for j in range(i):
            ej=np.eye(5)[j]*h
            H[i,j]=H[j,i]=(objective(z+ei+ej)-objective(z+ei-ej)-objective(z-ei+ej)+objective(z-ei-ej))/(4*h*h)
    grid=np.linspace(max(-.999,z[0]-4*sd[0]),min(.999,z[0]+4*sd[0]),25); profile=[]
    for d in grid:
        rr=least_squares(lambda nn:adapter.residual(np.r_[d,nn]),z[1:],bounds=(-1.,1.),method='trf',jac='3-point',diff_step=1e-4,max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        profile.append({'normalized_d0':d,'physical_d0':space.decode(np.r_[d,rr.x])['d0'],'delta_nll':rr.cost-fit.cost,'converged':rr.success,'normalized_nuisance':rr.x})
    roots=[]
    for a,b in zip(profile[:-1],profile[1:]):
        if (a['delta_nll']-1.920729410347062)*(b['delta_nll']-1.920729410347062)<0:
            roots.append(a['physical_d0']+(b['physical_d0']-a['physical_d0'])*(1.920729410347062-a['delta_nll'])/(b['delta_nll']-a['delta_nll']))
    # Parametric bootstrap conditioned on fitted mechanics + ONE fitted shared registration.
    fitted=space.decode(z); bt=generate_truth(pattern,model,fitted,[0.,10.,20.],x,load_history=fatigue_loading(pattern)); bootstrap=[]
    for bs in range(c['calibration']['bootstrap_first_seed'],c['calibration']['bootstrap_first_seed']+c['calibration']['bootstrap_seeds']):
        bf=observation_series(pattern,bt,x,sigma_mm=c['calibration']['sigma_mm'],seed=bs,registration=RegistrationError([fitted['tx'],fitted['ty']],fitted['theta']))
        ba=PhysicsForwardAdapter(predictor,ObservationBatch(bf,'independent_given_shared_registration'))
        br=least_squares(ba.residual,z,bounds=(-1.,1.),jac='3-point',diff_step=1e-4,method='trf',max_nfev=120)
        bootstrap.append({'seed':bs,'converged':br.success,'physical':space.decode(br.x),'normalized':br.x})
    errors=np.array([r['physical_parameter_error'] for r in records if 'parameter_names' in r]); widths=np.array([r['parameter_sd'] for r in records if 'parameter_names' in r])
    diagnostic={'parameter_names':space.names,'shared_nuisance_dimension':5,'frame_count':3,'full_cross_block_norm':np.linalg.norm(F[:2,2:]),'cross_block':F[:2,2:],'correlation':corr,'gauss_newton_fisher':F,'observed_hessian':H,'curvature_relative_norm':np.linalg.norm(H-F)/np.linalg.norm(F),'observed_hessian_eigenvalues':np.linalg.eigvalsh(H),'laplace_sd_physical':sd*space.scales,'observed_hessian_sd_physical':np.sqrt(np.diag(np.linalg.inv(H)))*space.scales,'boundary_margin_normalized':1-np.abs(z),'profile':profile,'profile_d0_interval':roots,'local_d0_interval':[fitted['d0']-1.95996398454*sd[0]*space.scales[0],fitted['d0']+1.95996398454*sd[0]*space.scales[0]],'bootstrap':bootstrap,'bootstrap_d0_percentile_interval':np.quantile([b['physical']['d0'] for b in bootstrap],[.025,.975]),'bootstrap_precision':'16 draws exploratory, not coverage validation','error_to_sd_RMS':np.sqrt(np.mean((errors/widths)**2,axis=0)),'covariance_contract':'independent frame measurement noise GIVEN one shared tx/ty/theta; registration NOT also inserted into R','interval_contract':'parameter marginal interval; future damage and predictive NIS are separate','qualification_change':False}
    (out/'diagnostic.json').write_text(json.dumps(clean(diagnostic),indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(clean({'independent_replication':summary,'wall_seconds':time.perf_counter()-t,'original_d0_coverage':89/100,'original_wilson95':[.8136870349691969,.9374580364293543],'seeds_frozen_before_execution':seeds}),indent=2)+'\n')
    print(json.dumps(clean(summary),indent=2))

if __name__=='__main__':main()
