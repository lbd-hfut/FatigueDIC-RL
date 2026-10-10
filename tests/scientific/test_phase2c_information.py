from dataclasses import replace
import numpy as np
import pytest
from fatigue_dic_rl.training.correction_dataset import generate_correction_case
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter
from fatigue_dic_rl.inference.belief_state import ObservationFrame,ObservationBatch
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian,posterior_uncertainty
from fatigue_dic_rl.observations.covariance import DisplacementCovariance
from fatigue_dic_rl.observations.noise import DICMeasurement
from fatigue_dic_rl.evaluation.correction_metrics import damage_metrics
from fatigue_dic_rl.core.validation import ContractError


@pytest.mark.scientific
def test_mask_covariance_and_DIC_permutation_preserve_likelihood(tmp_path):
    p=tmp_path/'case';generate_correction_case(p,seed=191,split='train',scenario='matched',sigma_mm=2e-7,scales=(1.,));a=load_correction_acquisition(p,mode='hybrid');predictor=a.predictors[0];frame=a.batches[0].frames[-1];m=frame.case.measurement;x=m.observation.coordinates;q=len(x)
    covariance=DisplacementCovariance.spatial(x,(2e-7)**2*np.array([[1.,.25],[.25,1.6]]),.22)
    mask=np.ones(q,dtype=bool);mask[::3]=False;m=DICMeasurement(m.observation,covariance,mask)
    f=ObservationFrame('masked',replace(frame.case,measurement=m));ad=MultiTrajectoryAdapter([predictor],[ObservationBatch((f,))]);z=predictor.parameter_space.encode({'d0':.12,'c0':.2,'c1':0.})
    u=ad.adapters[0].predict(z)[0][0];r=ad.residual(z);L=np.linalg.cholesky(covariance.selected_matrix(mask));np.testing.assert_allclose(r,np.linalg.solve(L,(u-m.observation.displacement)[mask].ravel()))
    perm=np.arange(q)[::-1];indices=np.column_stack((2*perm,2*perm+1)).ravel();o=replace(m.observation,coordinates=x[perm],displacement=m.observation.displacement[perm],valid_mask=m.observation.valid_mask[perm],confidence=m.observation.confidence[perm]);pm=DICMeasurement(o,DisplacementCovariance(covariance.matrix[np.ix_(indices,indices)]),mask[perm]);pf=ObservationFrame('permuted',replace(frame.case,measurement=pm));other=MultiTrajectoryAdapter([predictor],[ObservationBatch((pf,))]);rr=other.residual(z)
    assert r@r==pytest.approx(rr@rr,rel=2e-10)


@pytest.mark.scientific
def test_unknown_E0_static_damage_direction_remains_unidentified(tmp_path):
    from fatigue_dic_rl.inverse.parameterizations import ParameterSpace,Parameter,UniformDamage
    from fatigue_dic_rl.correction.physics_predictor import CorrectionPredictor
    p=tmp_path/'case';generate_correction_case(p,seed=192,split='train',scenario='matched',sigma_mm=0,scales=(1.,));a=load_correction_acquisition(p,included_cycles=(0.,));old=a.predictors[0]
    space=ParameterSpace((Parameter('d0',.02,.3,'1'),Parameter('E0',1500,2500,'N/mm^2')),{'A':8e-5,'m':1.})
    predictor=CorrectionPredictor(UniformDamage(),space,old.pattern,load_history=old.load_history);ad=MultiTrajectoryAdapter([predictor],[a.batches[0]]);z=space.encode({'d0':.12,'E0':2000.});s,m=checked_local_jacobian(ad.whitened_prediction,z);u=posterior_uncertainty(s,space.scales,normalized=z,derivative_method=m)
    assert u.data_rank==1 and u.covariance_normalized is None


def test_weighted_error_rejects_invalid_support():
    assert damage_metrics([0.,.2],[0.,0.],[3.,1.])['rmse']==pytest.approx(.1)
    for w in ([0.,1.],[np.nan,1.]):
        with pytest.raises(ContractError):damage_metrics([0.,.2],[0.,0.],w)
