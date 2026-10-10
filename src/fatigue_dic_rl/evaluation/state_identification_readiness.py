"""Small Development-only integrity, acceleration and cost experiments.

Truth is confined to generation and final error diagnostics. Inversion reads the
same observation-only artifacts for every information arm and both adapters.
No fatigue-law qualification, formal test worlds or uncertainty calibration.
"""
from dataclasses import replace, asdict
from pathlib import Path
import hashlib, json, time, tarfile
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.acquisition import public_patterns
from fatigue_dic_rl.correction.designed_acquisition import covariance_at
from fatigue_dic_rl.correction.robust_estimation import clean
from fatigue_dic_rl.correction.model_discriminability import project_whitened_difference
from fatigue_dic_rl.correction.information_sufficiency import BoundedProfileAdapter
from fatigue_dic_rl.evaluation.tri3_meshes import rectangle_mesh
from fatigue_dic_rl.inference.state_identification import load_static_cases
from fatigue_dic_rl.inference.uniform_static_scaling import UniformStaticScalingAdapter
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian, posterior_uncertainty
from fatigue_dic_rl.inverse.forward_adapter import StaticForwardAdapter
from fatigue_dic_rl.inverse.parameterizations import Parameter, ParameterSpace, UniformDamage
from fatigue_dic_rl.inverse.objectives import InverseObjective
from fatigue_dic_rl.inverse.optimizer import invert
from fatigue_dic_rl.inverse.sensitivity import SensitivityResult
from fatigue_dic_rl.inverse.forward_adapter import ReactionObservation
from fatigue_dic_rl.mechanics import MaterialAssignment
from fatigue_dic_rl.mechanics.tri3 import Tri3MechanicsSolver
from fatigue_dic_rl.observations.noise import simulate_measurement
from fatigue_dic_rl.observations.operator import ObservationOperator
from fatigue_dic_rl.observations.covariance import DisplacementCovariance


ARMS = (("DIC", False, False), ("DIC_reaction", True, False),
        ("DIC_healthy", False, True), ("DIC_reaction_healthy", True, True))


class TimedMechanics(Tri3MechanicsSolver):
    def __init__(self):
        super().__init__(); self.calls = 0; self.seconds = []

    def solve(self, request):
        self.calls += 1; start = time.perf_counter()
        try:
            return super().solve(request)
        finally:
            self.seconds.append(time.perf_counter()-start)


def space(config):
    return ParameterSpace(tuple(Parameter(n, *config[b], u) for n, b, u in (
        ("d0", "damage_bounds", "1"), ("E0", "modulus_bounds", "N/mm^2"),
        ("load_factor", "load_factor_bounds", "1"))), {})


def coordinates(q, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.uniform([.05, .03], [1.95, .97], (q, 2))
    if kind == "clustered":
        return rng.uniform([.08, .35], [.28, .65], (q, 2))
    n = int(np.sqrt(q))
    if n*n != q or kind != "regular":
        raise ContractError("regular probe sampling requires square Q")
    x, y = np.meshgrid(np.linspace(.05, 1.95, n), np.linspace(.03, .97, n))
    return np.column_stack((x.ravel(), y.ravel()))


def metadata(mesh, config, identity):
    return dict(schema="state-readiness-N0-v1", case_id=identity, split="development",
        mesh=dict(nodes=mesh.nodes, connectivity=mesh.connectivity,
            material_region_ids=mesh.material_region_ids, mesh_id=mesh.mesh_id,
            boundary_tags={k:dict(node_ids=v.node_ids, facets=v.facets) for k,v in mesh.boundary_tags.items()}),
        material=dict(E=2050., nu=.3, thickness=.8), load_scales=config["load_scales"],
        healthy_certificate=dict(externally_known_damage=0., source="independent_external_condition"),
        noise_contract="DIC frames independent conditional on shared physical parameters; sensor random errors independent, one shared sensor bias; healthy modulus not certified equal",
        units=dict(length="mm", force="N", modulus="N/mm^2"))


def candidate_request(pattern, physical, *, healthy=False):
    E = physical["E0"]; f = physical["load_factor"]
    mats = {r:MaterialAssignment(a.constitutive_model, dict(a.parameters,
        young_modulus_n_per_mm2=E)) for r,a in pattern.materials.items()}
    damage = np.zeros(pattern.quadrature_layout.count) if healthy else np.full(
        pattern.quadrature_layout.count, physical["d0"])
    return replace(pattern, materials=mats, damage=pattern.quadrature_layout.field(damage, "1"),
        traction_bcs=tuple(replace(b, values=b.values*f) for b in pattern.traction_bcs),
        nodal_force_bcs=tuple(replace(b, values=b.values*f) for b in pattern.nodal_force_bcs))


def generate_world(folder, scenario, seed, config):
    """Privileged synthetic generation; never imported by observation-only inference."""
    folder.mkdir(parents=True, exist_ok=False); rng = np.random.default_rng(seed)
    truth = dict(d0=float(rng.uniform(.08,.16)), E0=float(rng.uniform(1800.,2200.)),
                 load_factor=float(rng.uniform(.95,1.08)))
    mesh = rectangle_mesh(2,1,width_mm=2.,height_mm=1.)
    meta = metadata(mesh, config, folder.name)
    (folder/"public.json").write_text(json.dumps(clean(meta),indent=2)+"\n")
    patterns = public_patterns(meta); xy = coordinates(scenario["Q"],scenario["sampling"],seed+100)
    covariance = covariance_at(xy,scenario["sigma_mm"],scenario["spatial"])
    if scenario["spatial"]:
        fraction=config['spatial_independent_variance_fraction']
        if not 0 < fraction <= 1:
            raise ContractError('declared spatial/independent measurement mixture required')
        cov=covariance.matrix
        covariance=DisplacementCovariance((1-fraction)*cov+fraction*np.diag(np.diag(cov)))
    mechanics = TimedMechanics(); fields = {}
    for kind in ("damaged", "healthy"):
        for i, pattern in enumerate(patterns):
            physical = dict(truth)
            if kind == "healthy":
                physical["E0"] *= 1+scenario["healthy_E_relative_difference"]
            result = mechanics.solve(candidate_request(pattern,physical,healthy=kind=="healthy"))
            mapping = ObservationOperator(mesh).prepare(xy,coordinate_units="mm",coordinate_system="specimen")
            exact = mapping.apply(result.state.displacement).values
            measurement = simulate_measurement(xy,exact,covariance,seed=seed+1000+10*i+(kind=="healthy"),load_id=pattern.loading_state.load_id)
            target = folder/kind/str(i); measurement.save(target/"dic")
            if kind == "damaged":
                ids = mesh.boundary_tags["left"].node_ids
                force = float(result.reaction_forces.values[ids,0].sum())
                std = config["reaction_relative_std"]*abs(force/truth["load_factor"])
                observed = force*(1+scenario["reaction_sensor_relative_bias"])+rng.normal(0.,std)
                (target/"reaction.json").write_text(json.dumps(clean(dict(value_n=observed,
                    standard_deviation_n=std,node_ids=ids,component=0)),indent=2)+"\n")
            fields[kind+"_u_"+str(i)] = result.state.displacement.values
    # Truth stays in a separate evaluation directory, not the acquisition schema.
    private = folder.parent.parent/"synthetic_evaluation"/folder.name
    private.mkdir(parents=True,exist_ok=False)
    (private/"truth.json").write_text(json.dumps(dict(parameters=truth,scenario=scenario,seed=seed,
        externally_known_healthy_damage=0.,healthy_modulus=truth["E0"]*(1+scenario["healthy_E_relative_difference"])),indent=2)+"\n")
    np.savez_compressed(private/"fields.npz",**fields)
    return dict(FEM_solves=mechanics.calls,FEM_seconds=sum(mechanics.seconds))


def fit_with_cost(adapter, starts, config, *, max_nfev=None):
    start = time.perf_counter(); before = adapter.mechanics.calls
    result = invert(InverseObjective(adapter),starts,max_nfev=max_nfev or config["max_nfev"])
    attempts = list(result.starts); retries = []
    # Predetermined resource-based retry, independent of parameter truth or error.
    if not any(r.converged for r in attempts) and max_nfev is None:
        finite = [r for r in attempts if np.isfinite(r.cost)]
        if finite:
            initial = min(finite,key=lambda r:r.cost).normalized
            second = invert(InverseObjective(adapter),[initial],max_nfev=config["retry_max_nfev"])
            retries = list(second.starts); attempts.extend(retries)
    opt_seconds = time.perf_counter()-start; opt_calls = adapter.mechanics.calls-before
    valid = [r for r in attempts if np.isfinite(r.cost)]
    if not valid:
        return dict(status="FAILED_RETAINED",attempts=[asdict(r) for r in attempts],
                    optimizer_FEM_calls=opt_calls,optimizer_seconds=opt_seconds)
    selected = min([r for r in valid if r.converged] or valid,key=lambda r:r.cost)
    z = selected.normalized; t = time.perf_counter(); before = adapter.mechanics.calls
    sensitivity, method = checked_local_jacobian(adapter.whitened_prediction,z)
    info = posterior_uncertainty(sensitivity,adapter.parameter_space.scales,normalized=z,
        converged=selected.converged,derivative_method=method)
    jac_seconds = time.perf_counter()-t; jac_calls = adapter.mechanics.calls-before
    conditional = project_whitened_difference(info.jacobian_whitened[:,0],
        info.jacobian_whitened[:,1:],numerical_resolution=5*info.finite_difference_error)
    residual = adapter.residual(z); physical = adapter.parameter_space.decode(z)
    return dict(status="CONVERGED" if selected.converged else "UNCONVERGED_RETAINED",
        normalized=z,parameters=physical,data_cost=float(.5*residual@residual),
        rank=info.data_rank,singular_values=info.singular_values_data,
        weak_directions=info.weak_directions,null_directions=info.data_null_directions,
        conditional_damage_information=float(np.linalg.norm(conditional.projected_difference)),
        damage_identification="ABSTAIN" if np.linalg.norm(conditional.projected_difference)<3 else "CONDITIONAL_ON_HEALTH_TRANSFER_MODEL",
        boundary=bool(np.any(np.abs(z)>1-1e-6)),uncertainty_status=info.status,
        covariance_qualification="NOT_CALIBRATED; unmodeled healthy transfer and sensor bias excluded",
        attempts=[asdict(r) for r in attempts], retry_count=len(retries),
        optimizer_nfev=[r.nfev for r in attempts],optimizer_residual_calls=[len(r.history) for r in attempts],
        optimizer_FEM_calls=opt_calls,optimizer_seconds=opt_seconds,
        explicit_Jacobian_Fisher_FEM_calls=jac_calls,explicit_Jacobian_Fisher_seconds=jac_seconds,
        FD_checked=sensitivity.converged,J=info.jacobian_whitened,F=info.data_fisher,
        residual=residual,displacement=adapter.displacement_prediction(z))


def verification(cases, ps, config):
    """Declared tolerances; original fields and centered FD are the reference."""
    original = StaticForwardAdapter(UniformDamage(),ps,cases,mechanics=TimedMechanics())
    scaled = UniformStaticScalingAdapter(UniformDamage(),ps,cases,mechanics=TimedMechanics())
    rng = np.random.default_rng(282900); errors = []; jac_errors = []; residual_errors = []
    for z in [np.zeros(3), *rng.uniform(-.8,.8,(8,3))]:
        physical = ps.decode(z)
        for i,c in enumerate(scaled.cases):
            r = original.mechanics.solve(candidate_request(c.pattern,physical,healthy=c.known_healthy))
            expected = dict(displacement=r.state.displacement.values,strain=r.state.strain.values,
                stress=r.state.stress.values,reaction=r.reaction_forces.values,
                applied_force=r.applied_forces.values,elastic_strain_energy=r.elastic_strain_energy)
            fields = scaled.scaled_fields(z,i)
            for key in fields:
                a,b=np.asarray(fields[key]),np.asarray(expected[key])
                np.testing.assert_allclose(a,b,rtol=config["relative_field_tolerance"],atol=config["absolute_field_tolerance"])
                errors.append(float(np.max(np.abs(a-b))))
        a,b=original.residual(z),scaled.residual(z)
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-8)
        residual_errors.append(float(np.max(np.abs(a-b))))
        fd,_=checked_local_jacobian(original.whitened_prediction,z)
        exact=scaled.analytic_whitened_jacobian(z)
        relative=float(np.linalg.norm(fd.jacobian-exact)/max(np.linalg.norm(exact),1e-12))
        assert relative<=config["jacobian_relative_tolerance"];jac_errors.append(relative)
    return dict(max_field_absolute_error=max(errors),max_whitened_residual_difference=max(residual_errors),
        max_J_relative_error=max(jac_errors),candidates=9,reference_FEM_calls=original.mechanics.calls,
        reference_setup_FEM_calls=scaled.mechanics.calls,passed=True,
        optimizer_uses="existing TRF 3-point finite differences; analytic Jacobian is independently verified, not substituted into TRF")


def single_solve_cost(config):
    rows=[]
    for nx,ny in ((2,1),(4,2),(8,4)):
        mesh=rectangle_mesh(nx,ny,width_mm=2.,height_mm=1.)
        p=public_patterns(metadata(mesh,config,"timing"))[1];mech=TimedMechanics()
        for _ in range(20):mech.solve(p)
        rows.append(dict(nodes=mesh.n_nodes,elements=len(mesh.connectivity),calls=20,
            median_seconds=float(np.median(mech.seconds)),p95_seconds=float(np.quantile(mech.seconds,.95))))
    return rows


def health_transfer_diagnostics(cases,ps):
    """Exact nonideal-transfer counterexamples and prospective information only."""
    ad=UniformStaticScalingAdapter(UniformDamage(),ps,cases,mechanics=TimedMechanics())
    z=np.zeros(3);j=ad.analytic_whitened_jacobian(z);rho=[];gain=[];sensor=[]
    for i,(c,factor) in enumerate(zip(ad.cases,ad.factors)):
        u,r=ad.predict(z)[i];white=factor.whiten(u[c.measurement.active_mask].ravel())
        rho.extend(-.03*white if c.known_healthy else np.zeros(len(white)))
        gain.extend(.01*white if c.known_healthy else np.zeros(len(white)))
        sensor.extend(np.zeros(len(white)))
        if r is not None:
            rho.append(0.);gain.append(.01*r/c.reaction.standard_deviation_n if c.known_healthy else 0.)
            sensor.append(.03*r/c.reaction.standard_deviation_n)
    def information(extra):
        jac=np.column_stack((j,*extra));k=jac.shape[1]
        sensitivity=SensitivityResult(jac,jac,np.ones(k)*1e-3,np.zeros(k),1e-9,True)
        u=posterior_uncertainty(sensitivity,np.ones(k),normalized=np.zeros(k))
        d=project_whitened_difference(jac[:,0],jac[:,1:],numerical_resolution=1e-8)
        return dict(rank=u.data_rank,dimension=k,singular_values=u.singular_values_data,
                    conditional_damage_information=float(np.linalg.norm(d.projected_difference)))
    info=dict(equal_modulus_equal_loading=information([]),
        unknown_healthy_modulus_ratio=information([rho]),
        unknown_relative_healthy_loading=information([gain]),
        unknown_shared_force_sensor_bias=information([sensor]),
        both_transfer_nuisances=information([rho,gain]))
    # A future design, not an acquired healthy reaction or an inference result.
    prospective=[]
    for i,c in enumerate(cases):
        if c.known_healthy:
            base=cases[i-len(cases)//2].reaction
            c=replace(c,reaction=ReactionObservation(base.node_ids,base.component,0.,base.standard_deviation_n))
        prospective.append(c)
    both=UniformStaticScalingAdapter(UniformDamage(),ps,prospective,mechanics=TimedMechanics())
    jb=both.analytic_whitened_jacobian(z);gb=[]
    for i,(c,factor) in enumerate(zip(both.cases,both.factors)):
        u,r=both.predict(z)[i];white=factor.whiten(u[c.measurement.active_mask].ravel())
        gb.extend(.01*white if c.known_healthy else np.zeros(len(white)))
        if r is not None:gb.append(.01*r/c.reaction.standard_deviation_n if c.known_healthy else 0.)
    jold=j;j=jb
    info['prospective_both_specimen_reactions_unknown_relative_loading']=information([gb]);j=jold
    physical=dict(d0=.12,E0=2050.,load_factor=1.)
    pattern=ad.cases[0].pattern;mech=TimedMechanics()
    actual_damage=mech.solve(candidate_request(pattern,physical))
    actual_healthy=mech.solve(candidate_request(pattern,physical,healthy=True))
    ridge=[]
    for damage in (.10,.12,.14):
        modulus=physical['E0']*(1-physical['d0'])/(1-damage)
        relative=physical['E0']/modulus-1
        candidate=dict(d0=damage,E0=modulus,load_factor=1.)
        state=mech.solve(candidate_request(pattern,candidate))
        health=mech.solve(candidate_request(pattern,dict(candidate,E0=modulus*(1+relative)),healthy=True))
        ridge.append(dict(parameters=candidate,healthy_E_relative_difference=relative,
            within_declared_three_percent_transfer_bound=bool(abs(relative)<=.03),
            max_damage_displacement_difference=float(np.max(np.abs(state.state.displacement.values-actual_damage.state.displacement.values))),
            max_healthy_displacement_difference=float(np.max(np.abs(health.state.displacement.values-actual_healthy.state.displacement.values))),
            max_reaction_difference=float(np.max(np.abs(state.reaction_forces.values-actual_damage.reaction_forces.values)))))
    transfer=[]
    for relative in (-.03,-.01,0.,.01,.03):
        damage=1-(1-physical['d0'])/(1+relative)
        transfer.append(dict(healthy_E_relative_difference=relative,
            noiseless_shared_E_fit_damage=damage,systematic_damage_error=damage-physical['d0']))
    return clean(dict(local_information=info,bounded_actual_FE_equivalent_solutions=ridge,
        noiseless_transfer_bias=transfer,FEM_calls=ad.mechanics.calls+both.mechanics.calls+mech.calls,
        scope='analytic/numerical mechanism diagnostics and prospective design; no new independent worlds or calibration',
        healthy_D_zero_certificate_does_not_certify_equal_E=True))


def profile_cost_probe(folder,ps,config):
    """Bounded likelihood costs, not confidence intervals or calibration claims."""
    rows=[]
    for name,reaction in (('DIC_healthy',False),('DIC_reaction_healthy',True)):
        cases=load_static_cases(folder,reaction=reaction,healthy=True)
        for label,cls in (('original',StaticForwardAdapter),('exact_scaling',UniformStaticScalingAdapter)):
            base=cls(UniformDamage(),ps,cases,mechanics=TimedMechanics());base.evidence_ids=tuple(range(len(cases)))
            for value in config['development_profile_damage_grid']:
                adapter=BoundedProfileAdapter(base,'d0',value)
                before=base.mechanics.calls;start=time.perf_counter()
                result=invert(InverseObjective(adapter),[np.zeros(2),np.array([.4,-.35])],max_nfev=config['max_nfev'])
                finite=[r for r in result.starts if np.isfinite(r.cost)]
                selected=min([r for r in finite if r.converged] or finite,key=lambda r:r.cost) if finite else None
                rows.append(dict(arm=name,adapter=label,fixed_damage=value,
                    data_cost=selected.cost if selected else None,converged=bool(selected and selected.converged),
                    FEM_calls=base.mechanics.calls-before,seconds=time.perf_counter()-start,
                    setup_FEM_calls=len(cases) if label=='exact_scaling' and value==config['development_profile_damage_grid'][0] else 0,
                    attempts=[asdict(r) for r in result.starts]))
    checks=[]
    for arm in ('DIC_healthy','DIC_reaction_healthy'):
        for value in config['development_profile_damage_grid']:
            a,b=[next(r for r in rows if r['arm']==arm and r['adapter']==label and r['fixed_damage']==value) for label in ('original','exact_scaling')]
            error=abs(a['data_cost']-b['data_cost'])/max(1.,a['data_cost'])
            assert error<=config['fit_relative_cost_tolerance']
            checks.append(dict(arm=arm,damage=value,relative_cost_error=error,passed=True))
    return clean(dict(points=rows,cost_verification=checks,
        scope='five bounded damage values, two starts, one Development world; profile costs verified, unidentifiable nuisance parameter representatives not required equal'))


def run_readiness(output,config,config_sha):
    root=Path(output)
    if len(config["world_seeds"])!=len(config["worlds"]) or len(config["worlds"])>6:
        raise ContractError("this entry point refuses a formal or expanded world matrix")
    if config["scope"]!="initial_uniform_static_damage_development_only":
        raise ContractError("Development-only scope required")
    root.mkdir(parents=True,exist_ok=False);t=time.perf_counter();ps=space(config)
    paths=sorted(p for directory in ('src','scripts','tests','configs') for p in Path(directory).rglob('*')
                 if p.is_file() and p.suffix in ('.py','.toml') and '__pycache__' not in p.parts)
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with tarfile.open(root/'runtime_source.tar.gz','w:gz') as archive:
        for p in paths:archive.add(p,arcname=str(p),recursive=False)
    (root/'frozen_plan.json').write_text(json.dumps(dict(config=config,config_sha256=config_sha,
        runtime_source_manifest=source,independent_worlds=6,formal_test=False,
        point_error_scope='tentative optimizer coordinates are not unique damage estimates when ABSTAIN'),indent=2)+'\n')
    costs=single_solve_cost(config);worlds=[];checks=[]
    for index,(seed,scenario) in enumerate(zip(config['world_seeds'],config['worlds'])):
        identity=f'development-{index:02d}';folder=root/'observations'/identity
        gen=generate_world(folder,scenario,seed,config);arms={}
        for name,reaction,healthy in ARMS:
            cases=load_static_cases(folder,reaction=reaction,healthy=healthy)
            pair={};adapters={}
            for label,cls in (('original',StaticForwardAdapter),('exact_scaling',UniformStaticScalingAdapter)):
                if index not in (0,3) and label=='original':continue
                init=time.perf_counter();mech=TimedMechanics();ad=None
                try:
                    ad=cls(UniformDamage(),ps,cases,mechanics=mech)
                    record=fit_with_cost(ad,config['starts'],config)
                except Exception as exc:
                    record=dict(status='FAILED_RETAINED',reason=type(exc).__name__+': '+str(exc),
                                optimizer_FEM_calls=0,optimizer_seconds=0.)
                record.update(total_FEM_calls=mech.calls,FEM_seconds=sum(mech.seconds),
                    reference_setup_FEM_calls=mech.calls-record['optimizer_FEM_calls']-record.get('explicit_Jacobian_Fisher_FEM_calls',0),
                    total_arm_seconds=time.perf_counter()-init,adapter_prediction_evaluations=ad.evaluation_count if ad else 0,
                    DIC_scalars=sum(2*int(c.measurement.active_mask.sum()) for c in cases),
                    reactions=sum(c.reaction is not None for c in cases))
                pair[label]=record;adapters[label]=ad
            if 'original' in pair and pair['original']['status']!='FAILED_RETAINED' and pair['exact_scaling']['status']!='FAILED_RETAINED':
                a,b=pair['original'],pair['exact_scaling']
                delta=abs(a['data_cost']-b['data_cost'])/max(1.,a['data_cost'])
                assert delta<=config['fit_relative_cost_tolerance']
                if healthy:
                    assert abs(a['parameters']['d0']-b['parameters']['d0'])<=config['fit_damage_tolerance_identifiable']
                if healthy and reaction:
                    np.testing.assert_allclose(a['normalized'],b['normalized'],rtol=0,atol=2e-5)
                np.testing.assert_allclose(a['displacement'],b['displacement'],rtol=2e-5,atol=2e-8)
                checks.append(dict(world=identity,arm=name,relative_cost_difference=delta,
                    damage_difference=abs(a['parameters']['d0']-b['parameters']['d0']),
                    max_displacement_difference=float(np.max(np.abs(a['displacement']-b['displacement']))),
                    full_parameter_equality_required=bool(healthy and reaction),passed=True))
            target=root/'fits'/identity/name;target.mkdir(parents=True,exist_ok=False)
            for label,record in pair.items():
                arr={key:record.pop(key) for key in ('J','F','residual','displacement') if key in record}
                np.savez_compressed(target/(label+'.npz'),**arr)
                (target/(label+'.json')).write_text(json.dumps(clean(record),indent=2)+'\n')
            arms[name]=pair
        # Independent evaluation after all inference; truth never selects a fit.
        truth=json.loads((root/'synthetic_evaluation'/identity/'truth.json').read_text())
        for arm in arms.values():
            for record in arm.values():
                if 'parameters' in record:
                    record['evaluation_only_damage_error']=record['parameters']['d0']-truth['parameters']['d0']
                    record['evaluation_only_load_error']=record['parameters']['load_factor']-truth['parameters']['load_factor']
        worlds.append(dict(identity=identity,seed=seed,scenario=scenario,generation_cost=gen,arms=arms,
                           truth_evaluation_only=truth['parameters']))
        print(identity,'completed four paired observation arms',flush=True)
    cases=load_static_cases(root/'observations/development-00',reaction=True,healthy=True)
    fields=verification(cases,ps,config)
    # Explicit one-evaluation failure then retry: retained resource-failure evidence.
    retry_ad=UniformStaticScalingAdapter(UniformDamage(),ps,cases,mechanics=TimedMechanics())
    negative=fit_with_cost(retry_ad,[[-.8,-.6,.7]],config,max_nfev=config['negative_budget_probe_max_nfev'])
    retry=fit_with_cost(retry_ad,[negative.get('normalized',[-.8,-.6,.7])],config)
    negative.pop('J',None);negative.pop('F',None);negative.pop('residual',None);negative.pop('displacement',None)
    retry.pop('J',None);retry.pop('F',None);retry.pop('residual',None);retry.pop('displacement',None)
    # Same candidate repeated: the existing adapter cache is legal and observable.
    cache_ad=StaticForwardAdapter(UniformDamage(),ps,cases,mechanics=TimedMechanics())
    cache_ad.residual(np.zeros(3));first=cache_ad.mechanics.calls
    for _ in range(10):cache_ad.residual(np.zeros(3))
    repeated=cache_ad.mechanics.calls-first
    cache_ad.residual(np.array([.1,0,0]));changed=cache_ad.mechanics.calls-first-repeated
    result=dict(scope='DEVELOPMENT_ONLY; NOT independent validation',independent_worlds=6,
        paired_arms_per_world=4,worlds=worlds,single_solve=costs,field_verification=fields,
        fit_verification=checks,resource_failure_probe=negative,retry_probe=retry,
        existing_cache=dict(first_FEM_calls=first,repeated_ten_FEM_calls=repeated,changed_candidate_FEM_calls=changed),
        formal_test_executed=False,cycle_exposure=0,old_physics_modified=False,
        statistical_coverage_qualification='NOT EXECUTED',wall_seconds=time.perf_counter()-t)
    result['health_transfer']=health_transfer_diagnostics(cases,ps)
    result['profile_cost_probe']=profile_cost_probe(root/'observations/development-00',ps,config)
    result['wall_seconds']=time.perf_counter()-t
    write=time.perf_counter();(root/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n')
    files=[p for p in root.rglob('*') if p.is_file()]
    storage=dict(files=len(files),bytes=sum(p.stat().st_size for p in files),
                 summary_write_seconds=time.perf_counter()-write,
                 note='summary serialization/write measured; total workflow includes all artifact IO')
    (root/'storage.json').write_text(json.dumps(storage,indent=2)+'\n')
    return clean(result)
