from dataclasses import replace
import numpy as np
import pytest
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.observations.covariance import DisplacementCovariance
from fatigue_dic_rl.observations.registration import PixelCalibration, RegistrationError
from fatigue_dic_rl.observations.noise import DICMeasurement, simulate_measurement, calibrated_measurement


@pytest.mark.parametrize("matrix", [np.eye(3), np.ones((2, 3)), [[1, 2], [0, 1]], [[1, 0], [0, -1]], [[np.nan, 0], [0, 1]]])
def test_invalid_covariance_is_rejected(matrix):
    with pytest.raises(ContractError): DisplacementCovariance(matrix)


def test_covariance_units_required():
    with pytest.raises(ContractError): DisplacementCovariance(np.eye(2), "pixel^2")


@pytest.mark.parametrize("sigma", [[-1, 1], [[-1, 1]], [[np.inf, 1]], [1, 1], [[1, 1, 1]]])
def test_invalid_diagonal_covariance(sigma):
    with pytest.raises(ContractError): DisplacementCovariance.diagonal(sigma)


def test_block_selection_whitening_and_complete_likelihood():
    blocks = np.array([[[2., .4], [.4, 1.]], [[3., -.2], [-.2, 2.]]])
    cov = DisplacementCovariance.per_point(blocks)
    f = cov.factor(np.array([False, True]))
    e = np.array([.7, -.2])
    np.testing.assert_allclose(f.whiten(e) @ f.whiten(e), e @ np.linalg.solve(blocks[1], e), atol=1e-15)
    expected = .5*(e @ np.linalg.solve(blocks[1], e)+np.linalg.slogdet(blocks[1])[1]+2*np.log(2*np.pi))
    assert f.negative_log_likelihood(e) == pytest.approx(expected)
    with pytest.raises(ValueError): cov.matrix.setflags(write=True)


def test_singular_sampling_is_exact_and_likelihood_floor_is_explicit():
    cov = DisplacementCovariance(np.zeros((4, 4)))
    assert np.count_nonzero(cov.sample(np.random.default_rng(4), count=10)) == 0
    with pytest.raises(ContractError): cov.factor(np.ones(2, dtype=bool))
    f = cov.factor(np.ones(2, dtype=bool), variance_floor_mm2=.01)
    assert f.variance_floor_mm2 == .01
    np.testing.assert_allclose(f.whiten(np.ones(4)), 10)


@pytest.mark.parametrize("length", [0., -1., np.nan])
def test_spatial_length_must_be_positive(length):
    with pytest.raises(ContractError): DisplacementCovariance.spatial([[0, 0]], np.eye(2), length)


def test_spatial_kernel_matches_anisotropic_distance_formula():
    x = np.array([[0., 0.], [1., 0.], [.2, .7]])
    block = np.array([[2., .5], [.5, 1.]])
    c = DisplacementCovariance.spatial(x, block, .4)
    np.testing.assert_allclose(c.matrix[:2, 2:4], np.exp(-1/(2*.4**2))*block)
    assert np.linalg.eigvalsh(c.matrix)[0] > 0


@pytest.mark.parametrize("scale", [0., -1., np.inf])
def test_bad_calibration(scale):
    with pytest.raises(ContractError): PixelCalibration(scale)


def test_calibration_converts_displacement_and_covariance():
    c = PixelCalibration(.02, (.1, .2))
    o = calibrated_measurement([[10., 20.]], [[.4, -.2]], coordinate_units="pixel", displacement_units="pixel",
        calibration=c, covariance=np.diag([.01, .04]), covariance_units="pixel^2",
        valid_mask=[True], roi_mask=[True], confidence=[.2])
    np.testing.assert_allclose(o.observation.coordinates, [[.3, .6]])
    np.testing.assert_allclose(o.observation.displacement, [[.008, -.004]])
    np.testing.assert_allclose(o.covariance.matrix, np.diag([.01, .04])*.02**2)
    with pytest.raises(ContractError): c.displacement_to_mm([[1, 2]], "m")
    with pytest.raises(ContractError): c.covariance_to_mm2(np.eye(2), "pixel")


def test_registration_is_separate_from_random_noise_and_small_angle():
    x = np.array([[0., 0.], [1., 1.]])
    reg = RegistrationError([.1, .2], .01)
    cov = DisplacementCovariance(np.zeros((4, 4)))
    o = simulate_measurement(x, np.zeros_like(x), cov, seed=19, registration=reg)
    np.testing.assert_allclose(o.observation.displacement, [[.1, .2], [.09, .21]])
    assert np.count_nonzero(o.covariance.matrix) == 0
    with pytest.raises(ContractError): RegistrationError([0, 0], .1)


def test_missing_roi_reproducibility_and_no_truth_channels(tmp_path):
    x = np.random.default_rng(6).random((20, 2))
    c = DisplacementCovariance.diagonal(np.ones_like(x)*.01)
    a = simulate_measurement(x, x*.02, c, seed=31, missing_fraction=.3, roi_mask=np.arange(20) % 2 == 0)
    b = simulate_measurement(x, x*.02, c, seed=31, missing_fraction=.3, roi_mask=np.arange(20) % 2 == 0)
    np.testing.assert_array_equal(a.observation.displacement, b.observation.displacement)
    assert np.array_equal(a.active_mask, a.roi_mask & a.observation.valid_mask)
    a.save(tmp_path/"observations")
    loaded = DICMeasurement.load(tmp_path/"observations")
    np.testing.assert_array_equal(a.observation.displacement, loaded.observation.displacement)
    assert not hasattr(a, "damage") and not hasattr(a, "parameters")
    with pytest.raises(FileExistsError): a.save(tmp_path/"observations")
    with np.load(tmp_path/"observations"/"observation.npz") as data:
        values = dict(data)
    values["damage"] = np.zeros(20)
    np.savez(tmp_path/"observations"/"observation.npz", **values)
    with pytest.raises(ContractError): DICMeasurement.load(tmp_path/"observations")


def test_mm_wrapper_rejects_mixed_units_and_fake_masks():
    x = np.zeros((2, 2))
    cov = DisplacementCovariance(np.zeros((4, 4)))
    o = simulate_measurement(x, x, cov, seed=3)
    with pytest.raises(ContractError): DICMeasurement(replace(o.observation, displacement_units="m"), cov, [True, True])
    with pytest.raises(ContractError): simulate_measurement(x, x, cov, seed=3, domain_mask=[1, 0])


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_seed_contract(seed):
    with pytest.raises(ContractError): simulate_measurement([[0, 0]], [[0, 0]], DisplacementCovariance(np.eye(2)), seed=seed)
