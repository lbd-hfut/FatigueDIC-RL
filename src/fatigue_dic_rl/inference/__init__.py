from .estimator import StateEstimator

__all__ = ["StateEstimator"]


def __getattr__(name):
    # Preserve the NumPy-only Phase 0 import path. SciPy is loaded on demand.
    if name == "NeuralBeliefEstimator":
        from .neural_belief_estimator import NeuralBeliefEstimator
        return NeuralBeliefEstimator
    if name in {"PhysicsBasedBeliefEstimator", "SequentialMAPConfig"}:
        from . import sequential_map
        return getattr(sequential_map, name)
    if name in {"BeliefState", "MechanicsBelief", "ObservationBatch", "ObservationFrame",
                "ObservationDesign", "PredictedObservation", "EstimateRole", "GaussianPrior", "BeliefEstimator"}:
        from . import belief_state
        return getattr(belief_state, name)
    if name == "PhysicsPredictor":
        from .physics_predictor import PhysicsPredictor
        return PhysicsPredictor
    if name == "ForecastResult":
        from .forecast import ForecastResult
        return ForecastResult
    if name == "Innovation":
        from .innovation import Innovation
        return Innovation
    raise AttributeError(name)
