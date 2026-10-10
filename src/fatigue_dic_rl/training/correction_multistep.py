"""Observation-based multi-step calibration, explicitly finite-difference TRF."""
from fatigue_dic_rl.correction.acquisition import load_correction_acquisition
from fatigue_dic_rl.correction.correction_optimizer import MultiTrajectoryAdapter, fit_correction


def calibrate_prefix(folder, *, dimension=2, cycles=(0.,10.,20.), max_nfev=80, regularization=1e-3):
    acquisition=load_correction_acquisition(folder,mode='hybrid',dimension=dimension,included_cycles=cycles)
    adapter=MultiTrajectoryAdapter(acquisition.predictors,acquisition.batches)
    result=fit_correction(adapter,max_nfev=max_nfev,coefficient_regularization=regularization)
    return acquisition,adapter,result
