"""Cold-J, repeated-start, failure/retry and out-of-box Development cost probes."""
import argparse,json,time,tomllib
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from fatigue_dic_rl.inference.state_identification import load_static_cases
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian,posterior_uncertainty
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from fatigue_dic_rl.inverse.parameterizations import UniformDamage,ParameterSpace,Parameter
from fatigue_dic_rl.evaluation.state_identification_readiness import TimedMechanics,space,candidate_request
from fatigue_dic_rl.correction.robust_estimation import clean


def probe(folder,config):
    ps=space(config);cases=load_static_cases(folder,reaction=True,healthy=True)
    mech=TimedMechanics();cold=StaticForwardAdapter(UniformDamage(),ps,cases,mechanics=mech)
    start=time.perf_counter();s,kind=checked_local_jacobian(cold.whitened_prediction,np.zeros(3))
    u=posterior_uncertainty(s,ps.scales,normalized=np.zeros(3),derivative_method=kind)
    jac=dict(FEM_calls=mech.calls,seconds=time.perf_counter()-start,rank=u.data_rank,checked=s.converged)
    repeat=StaticForwardAdapter(UniformDamage(),ps,cases,mechanics=TimedMechanics());starts=[]
    for index,z in enumerate([config['starts'][0],config['starts'][1],config['starts'][0]]):
        before=repeat.mechanics.calls;t=time.perf_counter();r=invert(InverseObjective(repeat),[z],max_nfev=config['max_nfev']).starts[0]
        starts.append(dict(index=index,initial=z,FEM_calls=repeat.mechanics.calls-before,
            seconds=time.perf_counter()-t,nfev=r.nfev,residual_calls=len(r.history),
            result=asdict(r),repeated_first_initialization=index==2))
    failed=StaticForwardAdapter(UniformDamage(),ps,cases,mechanics=TimedMechanics())
    before=failed.mechanics.calls;t=time.perf_counter();r=invert(InverseObjective(failed),[[-.8,-.6,.7]],max_nfev=1).starts[0]
    failure=dict(FEM_calls=failed.mechanics.calls-before,seconds=time.perf_counter()-t,result=asdict(r))
    before=failed.mechanics.calls;t=time.perf_counter();r=invert(InverseObjective(failed),[r.normalized],max_nfev=config['retry_max_nfev']).starts[0]
    retry=dict(FEM_calls=failed.mechanics.calls-before,seconds=time.perf_counter()-t,result=asdict(r))
    # Separate engineering counterexample: physically legal D=.4 outside the
    # inverse box, known E/load; this adds no independent pilot world.
    c=cases[0];m=TimedMechanics();actual=m.solve(candidate_request(c.pattern,dict(d0=.4,E0=2050.,load_factor=1.)))
    u=cold.mappings[0].apply(actual.state.displacement).values
    measured=replace(c.measurement,observation=replace(c.measurement.observation,displacement=u))
    one=ParameterSpace((Parameter('d0',.02,.3,'1'),),dict(E0=2050.,load_factor=1.))
    boundary=StaticForwardAdapter(UniformDamage(),one,[replace(c,measurement=measured,reaction=None)],mechanics=TimedMechanics())
    t=time.perf_counter();b=invert(InverseObjective(boundary),[[0.]],max_nfev=60).starts[0]
    bound=dict(estimated_damage=one.decode(b.normalized)['d0'],boundary=bool(abs(b.normalized[0])>1-1e-6),
        optimizer_converged=b.converged,cost=b.cost,FEM_calls=boundary.mechanics.calls,
        seconds=time.perf_counter()-t,result=asdict(b),scope='noise-free synthetic out-of-inverse-box engineering test, not a successful recovery')
    return clean(dict(cold_Jacobian_Fisher=jac,per_initialization=starts,budget_failure=failure,retry=retry,
        boundary_probe=bound,independent_worlds_added=0,fatigue_cycles=0,formal_test=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observations',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();config=tomllib.loads(Path('configs/validation_readiness/development.toml').read_text())
    output=Path(a.output)
    if output.exists():raise RuntimeError('do not replace existing scientific evidence')
    output.write_text(json.dumps(probe(a.observations,config),indent=2)+'\n')
