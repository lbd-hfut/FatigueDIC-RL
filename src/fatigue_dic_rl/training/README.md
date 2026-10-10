# Synthetic-supervised amortized sequential inference

Case-level train/validation/calibration/test splits, direct verified FEM/fatigue
truth generation, AdamW point/NLL training, validation selection, five frozen
seeds and tensor-only checkpoint reload. Observation-only pseudo-label control
does not read internal damage labels. No replay buffer or RL trainer.

SciPy physics is stopped gradient. The final uniform causal training condition
uses a separately FEM-verified homogeneous scalar equivalent; production
inference/forecasts still use Phase 1B/FEM. No implicit/adjoint gradient claim.
Calibration and test cases are refused by the training loader.
See [neural contracts](../../../docs/12_NEURAL_BELIEF_CONTRACTS.md).
