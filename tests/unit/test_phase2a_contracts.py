import ast
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.inference.belief_state import (ObservationBatch, ObservationFrame, ObservationDesign, EstimateRole)
from fatigue_dic_rl.inference.physics_predictor import PhysicsForwardAdapter
from fatigue_dic_rl.inference.innovation import innovation
from fatigue_dic_rl.inference.sequential_map import SequentialMAPConfig, PhysicsBasedBeliefEstimator
from fatigue_dic_rl.evaluation.belief_metrics import TemporalSplit, ObservationFileRef, load_observation_prefix
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup, generate_truth, observation_series
from fatigue_dic_rl.observations.noise import DICMeasurement


@pytest.fixture
def frames():
    model, _, pattern, predictor, x = uniform_setup({"geometry": {"nx": 2, "ny": 1, "seed": 19}})
    ts = generate_truth(pattern, model, {"d0": .2}, [0., 10., 20.], x)
    return predictor, observation_series(pattern, ts, x, sigma_mm=5e-5)


def test_duplicate_chronology_and_unmodelled_frame_correlation_are_rejected(frames):
    _, fs = frames
    for values in ((fs[0], fs[0]), (fs[2], fs[0])):
        with pytest.raises(ContractError): ObservationBatch(values)
    with pytest.raises(ContractError): ObservationBatch(fs, "unknown_temporal_correlation")


def test_forecast_design_and_prediction_have_no_truth_or_measured_displacement(frames):
    predictor, fs = frames
    d = fs[0].design
    assert isinstance(d, ObservationDesign)
    assert not hasattr(d, "displacement") and not hasattr(d, "damage")
    assert d.coordinates_mm.shape == (16, 2)
    p = predictor.observe([0.], d)
    assert p.role == EstimateRole.PREDICTION and p.displacement_mm.shape == (16, 2)
    with pytest.raises(ValueError): p.displacement_mm[0, 0] = 1.
    with pytest.raises(ValueError): d.coordinates_mm.setflags(write=True)


@pytest.mark.parametrize("arguments", [{"max_nfev": 0}, {"max_nfev": True}, {"tolerance": 0.},
                                      {"assimilation_cutoff": -1.}, {"rank_rtol": np.nan}])
def test_invalid_sequential_configuration(arguments):
    with pytest.raises(ContractError): SequentialMAPConfig(**arguments)


def test_mask_and_spatial_covariance_match_independent_dense_quadratic_form(frames):
    predictor, fs = frames
    from fatigue_dic_rl.observations.covariance import DisplacementCovariance
    f = fs[0]; x = f.design.coordinates_mm
    c = DisplacementCovariance.spatial(x, np.array([[1., .3], [.3, 2.]])*1e-8, .2)
    roi = np.arange(len(x)) % 3 != 1
    m = DICMeasurement(f.case.measurement.observation, c, roi)
    masked = replace(f, case=replace(f.case, measurement=m))
    adapter = PhysicsForwardAdapter(predictor, ObservationBatch((masked,)))
    e = (adapter.predict([0.])[0][0]-m.observation.displacement)[m.active_mask].ravel()
    r = adapter.residual([0.])
    np.testing.assert_allclose(r @ r, e @ np.linalg.solve(c.selected_matrix(m.active_mask), e), rtol=1e-11)


def test_predictive_innovation_uses_R_plus_parameter_covariance_and_postfit_is_labelled(frames):
    predictor, fs = frames
    f = fs[0]
    p = predictor.observe([0.], f.design)
    j = np.full((32, 1), .1)
    c = np.array([[.02]])
    e = innovation(f, p, jacobian_whitened=j, covariance_normalized=c, posterior_reliable=True)
    expected = e.measurement_whitened @ np.linalg.solve(np.eye(32)+j @ c @ j.T, e.measurement_whitened)
    assert e.normalized_innovation_squared == pytest.approx(expected)
    fitted = replace(p, role=EstimateRole.FILTERED)
    e = innovation(f, fitted, jacobian_whitened=j, covariance_normalized=c, posterior_reliable=True)
    assert e.kind == "post_fit_residual" and e.normalized_innovation_squared is None


def test_loader_filters_future_before_file_access_and_rejects_truth_channels(tmp_path, frames, monkeypatch):
    predictor, fs = frames
    refs = []
    for i, f in enumerate(fs):
        dest = tmp_path/f"frame-{i}"
        f.case.measurement.save(dest)
        refs.append(ObservationFileRef(f.observation_id, f.cycle_count, dest, predictor.pattern))
    refs += [ObservationFileRef("future30", 30., tmp_path/"future-must-not-open", predictor.pattern),
             ObservationFileRef("future40", 40., tmp_path/"future-must-not-open-either", predictor.pattern)]
    loaded = []
    original = DICMeasurement.load
    def checked(folder):
        assert not Path(folder).name.startswith("future")
        loaded.append(folder)
        return original(folder)
    monkeypatch.setattr(DICMeasurement, "load", checked)
    batch = load_observation_prefix(refs, TemporalSplit())
    assert len(loaded) == 3 and batch.ids == tuple(f.observation_id for f in fs)
    dest = refs[0].folder/"observation.npz"
    with np.load(dest) as data: values = {k: data[k] for k in data.files}
    values["damage"] = np.zeros(4)
    np.savez(dest, **values)
    with pytest.raises(ContractError, match="unexpected channels"): original(refs[0].folder)


def test_inference_has_no_evaluation_truth_network_or_training_dependency():
    root = Path(__file__).resolve().parents[2]/"src"/"fatigue_dic_rl"/"inference"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not ({"evaluation", "truth", "networks", "rl", "training"} & set((node.module or "").split(".")))
        assert "SyntheticTruth" not in path.read_text()


def test_invalid_forecast_targets_are_rejected_and_duplicate_update_is_atomic(frames):
    predictor, fs = frames
    est = PhysicsBasedBeliefEstimator(predictor, initializations=[[0.]])
    est.assimilate_observation(fs[0])
    before = est.belief
    with pytest.raises(ContractError): est.assimilate_observation(fs[0])
    assert est.belief is before and len(est.attempts) == 1
    with pytest.raises(ContractError): est.forecast_future([fs[0].design])
    with pytest.raises(ContractError): est.forecast_future([fs[1]])
