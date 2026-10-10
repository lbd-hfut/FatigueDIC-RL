"""Physics-guided amortized inference implementing the Phase 2A protocol.

The injected backend owns torch/graph execution; this module imports neither
networks nor learning/truth data. This preserves classical inference isolation.
Candidate changes always replay a new consistent Phase 1B history from N=0.
"""
from dataclasses import dataclass,asdict,is_dataclass
import json,time
import numpy as np
from fatigue_dic_rl.core.validation import ContractError,array
from fatigue_dic_rl.fatigue.loading import LoadingMode
from .belief_state import MechanicsBelief,ObservationBatch,ObservationFrame,EstimateRole
from .forecast import predictions_from_belief,forecast_future
from .innovation import innovation


@dataclass(frozen=True)
class NeuralUncertainty:
    covariance_normalized: np.ndarray | None
    parameter_scales: np.ndarray
    status: str
    warnings: tuple[str,...]=()
    data_rank: int | None=None
    data_null_directions: np.ndarray | None=None
    data_fisher: np.ndarray | None=None

    def __post_init__(self):
        object.__setattr__(self,'parameter_scales',array(self.parameter_scales))
        c=self.covariance_normalized
        if c is not None:
            c=np.asarray(c);k=len(self.parameter_scales)
            if c.shape!=(k,k) or not np.isfinite(c).all() or not np.allclose(c,c.T,atol=1e-10) or np.linalg.eigvalsh(c).min()<=0:
                raise ContractError('learned covariance requires finite SPD full parameter support')
            object.__setattr__(self,'covariance_normalized',array(c))

    @property
    def covariance_physical(self):
        c=self.covariance_normalized;return None if c is None else array(c*np.outer(self.parameter_scales,self.parameter_scales))

    @property
    def marginal_standard_deviation_physical(self):
        c=self.covariance_physical;return np.full(len(self.parameter_scales),np.inf) if c is None else np.sqrt(np.diag(c))

    @property
    def reliable_for_propagation(self):
        # Numeric usability ONLY; the status never claims calibration/identifiability.
        return self.covariance_normalized is not None and 'active_parameter_bound' not in self.warnings


@dataclass(frozen=True)
class NeuralAssimilationResult:
    status: str
    prefix_ids: tuple[str,...]
    filtered_belief: MechanicsBelief | None
    predictions_before_update: tuple
    innovations_before_update: tuple
    normalized_before: tuple
    normalized_after: tuple
    elapsed_seconds: float
    diagnostics: tuple
    network_seconds: float=0.


class NeuralBeliefEstimator:
    def __init__(self,predictor,network,*,metadata=None,initial_normalized=None,assimilation_cutoff=20.,calibration=None,diagnose_identifiability=False,known_E0=True):
        self.predictor=predictor;self.network=network.eval();self.metadata=metadata or {};self.cutoff=float(assimilation_cutoff);self.calibration=calibration or {};self.diagnose_identifiability=diagnose_identifiability
        if not known_E0 or 'E0' in predictor.parameter_space.names:raise ContractError('unknown E0/damage ambiguity: unique neural damage inference unsupported')
        if predictor.parameter_space.names!=tuple(network.config.parameter_names):raise ContractError('checkpoint and physics parameter names disagree')
        if any(np.any(b.values!=0) for b in predictor.pattern.displacement_bcs):
            raise ContractError('checkpoint requires load control; imposed-displacement damage ambiguity is unsupported')
        if any(c.loading_mode!=LoadingMode.LOAD for c in predictor.load_history):
            raise ContractError('checkpoint trained for proportional load control, not displacement control')
        allowed=[('d0','A'),('d_base','amplitude','xc','yc','sx','sy')]
        if predictor.parameter_space.names not in allowed:raise ContractError('only declared low-dimensional trained parameter heads supported')
        if initial_normalized is None:
            initial_normalized=predictor.parameter_space.encode({'d0':.16,'A':.0015}) if predictor.parameter_space.names==('d0','A') else np.zeros(6)
        self.initial_normalized=np.asarray(initial_normalized);predictor.parameter_space.decode(self.initial_normalized)
        self.frames=();self.hidden=None;self.attempts=[];self._belief=None;self.initialize_belief()

    @property
    def belief(self):return self._belief

    def _make_belief(self,z,cycle,ids,role,uncertainty,latest_innovation=None):
        space=self.predictor.parameter_space;physical=space.decode(z);state=self.predictor.damage_states(z,[cycle])[cycle]
        provenance={'physics_model_id':self.predictor.model_id,'estimator':'neural_amortized_sequential_inference','network':self.metadata,'known_E0':True,'known_BC':True,'noise_contract':'conditionally independent frame noise given public mechanics; unmodelled bias is discrepancy','training_prior_is_not_DIC_information':True,'uncertainty_interpretation':uncertainty.status,'conditioned_ids':ids}
        return MechanicsBelief(space.names,z,np.array([physical[n] for n in space.names]),uncertainty,state.damage,cycle,role,ids,None,json.dumps(provenance,sort_keys=True),latest_innovation,'physics_at_neural_parameter_center_not_damage_posterior_mean')

    def initialize_belief(self):
        if self.frames:raise ContractError('cannot silently reset an assimilated neural belief')
        u=NeuralUncertainty(None,self.predictor.parameter_space.scales,'declared_training_prior_center_no_learned_covariance_yet')
        self._belief=self._make_belief(self.initial_normalized,0.,(),EstimateRole.PRIOR,u);return self._belief

    def estimate_initial(self,observations=None):return self.initialize_belief() if observations is None else self.assimilate_observation(observations)

    def predict_to_cycle(self,cycle_count,designs=(),*,propagate_uncertainty=False):
        return predictions_from_belief(self.predictor,self.belief,[cycle_count],designs,propagate_uncertainty=propagate_uncertainty)[0]

    def forecast_future(self,designs,*,propagate_uncertainty=True):
        return forecast_future(self.predictor,self.belief,designs,propagate_uncertainty=propagate_uncertainty)

    def assimilate_observation(self,observation):
        batch=observation if isinstance(observation,ObservationBatch) else ObservationBatch((observation,))
        if batch.noise_contract!='conditionally_independent':raise ContractError('this neural head has no shared registration nuisance parameters; explicit shared-noise contract unsupported')
        if set(batch.ids)&{f.observation_id for f in self.frames}:raise ContractError('duplicate evidence ID')
        if any(f.cycle_count>self.cutoff for f in batch.frames) or batch.frames[0].cycle_count<self.belief.cycle_count:raise ContractError('future observation or noncausal time rejected before update')
        for f in batch.frames:
            self.predictor.validate_design(f.design)
            o=f.case.measurement.observation
            if o.coordinate_system!=f.case.pattern.mesh.coordinate_system or o.loading_state_ref!=f.case.pattern.loading_state.load_id:
                raise ContractError('observation coordinates/loading reference disagrees with public physics')
        before_belief=self.belief;local=before_belief;hidden=self.network.clone_hidden(self.hidden);frames=list(self.frames);predictions=[];innovations=[];before=[];after=[];t=time.perf_counter();network_seconds=0.
        try:
            for frame in batch.frames:
                pre=predictions_from_belief(self.predictor,local,[frame.cycle_count],[frame.design])[0];e=innovation(frame,pre.observations[0]);predictions.append(pre);innovations.append(e);before.append(local.normalized_mean)
                output=self.network.observation_update(frame,self.predictor,local.normalized_mean,local.cycle_count,hidden)
                network_seconds+=output['network_seconds']
                z=output['normalized_mean'];hidden=output['hidden'];self.predictor.parameter_space.decode(z);after.append(z)
                covariance=output['covariance_normalized']
                status='learned_gaussian_uncalibrated' if covariance is not None else 'point_estimator_no_covariance';warnings=['training_prior_dependence_not_data_only_posterior']
                factor=self.calibration.get(str(float(frame.cycle_count)))
                if covariance is not None and factor is not None:
                    a=np.asarray(factor);covariance*=np.outer(a,a);status='learned_gaussian_calibration_split_rescaled'
                if np.any(np.abs(z)>=1-1e-6):warnings.append('active_parameter_bound')
                if covariance is not None and np.any(1-np.abs(z)<3*np.sqrt(np.diag(covariance))):warnings.append('Gaussian_mass_near_box_no_interval_clipping')
                frames.append(frame);rank=null=F=None
                if self.diagnose_identifiability:
                    from .physics_predictor import PhysicsForwardAdapter
                    from .uncertainty import checked_local_jacobian,posterior_uncertainty
                    adapter=PhysicsForwardAdapter(self.predictor,ObservationBatch(tuple(frames)));s,method=checked_local_jacobian(adapter.whitened_prediction,z);diagnostic=posterior_uncertainty(s,self.predictor.parameter_space.scales,normalized=z,derivative_method=method);rank=diagnostic.data_rank;null=diagnostic.data_null_directions;F=diagnostic.data_fisher
                    if rank<len(z):warnings.append('data_rank_deficient_learned_covariance_depends_on_training_prior')
                u=NeuralUncertainty(covariance,self.predictor.parameter_space.scales,status,tuple(warnings),rank,null,F)
                local=self._make_belief(z,frame.cycle_count,tuple(f.observation_id for f in frames),EstimateRole.FILTERED,u,e)
            self._belief=local;self.frames=tuple(frames);self.hidden=hidden
            result=NeuralAssimilationResult('SUCCESS',local.observation_ids,local,tuple(predictions),tuple(innovations),tuple(before),tuple(after),time.perf_counter()-t,(),network_seconds)
        except Exception as exc:
            diagnostic={'type':type(exc).__name__,'message':str(exc)}
            if hasattr(exc,'diagnostics') and is_dataclass(exc.diagnostics):diagnostic['mechanics']=asdict(exc.diagnostics)
            if hasattr(exc,'failures'):diagnostic['fatigue_failures']=[asdict(f) for f in exc.failures]
            result=NeuralAssimilationResult('FAILED_ROLLED_BACK',before_belief.observation_ids,None,tuple(predictions),tuple(innovations),tuple(before),tuple(after),time.perf_counter()-t,(diagnostic,),network_seconds)
        self.attempts.append(result);return result
