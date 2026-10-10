# Phase 2B graph/recurrent inference

Native torch DIC-coordinate kNN and Tri3 FE encoders, normalized shape-function
fusion or global pooling, causal graph-level GRU, bounded low-dimensional
parameter center and full Cholesky uncertainty head. No RL actor/critic/action.

Torch is optional and isolated from the NumPy/SciPy core. Public observations
and consistent candidate physics conditions are the only inference inputs.
FE nodes, elements, quadrature and DIC points have separate contracts.
See [neural contracts](../../../docs/12_NEURAL_BELIEF_CONTRACTS.md).
