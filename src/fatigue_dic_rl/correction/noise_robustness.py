"""Public observation diagnostics; displacement SNR is distinct from damage information."""
import numpy as np


def observation_diagnostics(acquisition,fit,adapter):
    prediction=adapter.displacement_prediction(fit.normalized)
    sigma=acquisition.metadata['sigma_mm']
    j=fit.uncertainty.jacobian_whitened
    names=adapter.parameter_space.names
    i=names.index('d0')
    amplitude=float(np.sqrt(np.mean(prediction**2)))
    scale=adapter.parameter_space.scales[i]
    dsignal=float(.01/scale*np.linalg.norm(j[:,i])/np.sqrt(j.shape[0]))
    return dict(displacement_feature_rms_mm=amplitude,noise_sigma_x_mm=sigma,
                displacement_snr=amplitude/sigma if sigma>0 else None,
                d0_change_0_01_whitened_signal_rms=dsignal,
                damage_sensitive_signal_below_component_noise=dsignal<1.,
                whitened_prefix_residual_rms=float(np.sqrt(np.mean(adapter.residual(fit.normalized)**2))),
                fisher_singular_values=fit.uncertainty.singular_values_data,
                numerical_rank=fit.uncertainty.data_rank,
                caveat='local box-scaled information; displacement SNR does not imply material-law identifiability')
