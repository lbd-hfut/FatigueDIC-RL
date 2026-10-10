"""Separate observational information from a numerical coefficient penalty."""
import numpy as np
from fatigue_dic_rl.inference.uncertainty import checked_local_jacobian,posterior_uncertainty


def correction_information(adapter,normalized,*,regularization=1e-3):
    sensitivity,method=checked_local_jacobian(adapter.whitened_prediction,normalized)
    data=posterior_uncertainty(sensitivity,adapter.parameter_space.scales,normalized=normalized,derivative_method=method)
    ridge=np.zeros(len(normalized))
    for j,name in enumerate(adapter.parameter_space.names):
        if name.startswith('c') and name[1:].isdigit():ridge[j]=regularization*adapter.parameter_space.scales[j]**2
    return data,np.diag(ridge)
