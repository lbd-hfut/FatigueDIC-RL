# Phase 1C — DIC and classical inverse mechanics contracts

This is an additive layer. Phase 0 DICObservation, Phase 1A mechanics and Phase
1B fatigue/history contracts retain their original behavior. Forward mechanics
does not import the inverse package. No neural networks or RL are used.

## Coordinates, displacement and covariance

`DICObservation` remains physical-coordinate, displacement-only data in m/mm.
`DICMeasurement` composes it with a full `DisplacementCovariance` and a separate
ROI mask, requiring physical mm at this new inverse boundary. Arbitrary finite
`[Q,2]` coordinates are supported; no image-grid reshape is required.

`PixelCalibration` converts pixel coordinates using an explicit positive
mm_per_pixel and physical origin. Displacements scale by mm_per_pixel; covariance
scales by its square. Units must be declared for all conversions. Converting
image coordinates does not solve general camera calibration, perspective or
stereo reconstruction. The first adapter assumes aligned axes and a scalar scale.

Active points = ROI AND validity. An outside point can be retained as invalid
with NaN displacement. An active outside point raises ContractError. Confidence
is descriptive acquisition information, not an inverse variance.

Covariance flattening is `[ux0,uy0,ux1,uy1,...]`, units mm². Diagonal, per-point
2×2 blocks and Gaussian distance-kernel full matrices are accepted. Symmetry and
PSD are checked with roundoff-scaled tolerances. PSD sampling uses an eigen root;
roundoff-negative eigenvalues alone are set to zero after validation. Zero noise
generates exact zero samples. Likelihood uses the selected covariance principal
submatrix, not the corresponding submatrix of a precision matrix.

Likelihood uses Cholesky whitening, `z=L^-1 e`, and
`NLL=.5*(z^T z+logdet(Sigma)+M log(2 pi))`. No full covariance inverse is formed.
A singular sampling covariance cannot define a nonsingular Gaussian likelihood.
Noiseless fitting explicitly declares an algorithmic variance floor (1e-10 mm²
in scientific recovery); it is not relabelled as physical measurement noise.
Noisy experiments use their actual covariance without an added floor.

`RegistrationError` is a separate additive translation and linear small-angle
rotation field about a declared center; |theta|≤0.05 rad. It is deterministic bias
in the current generator, not stochastic covariance. Generating values belong
in evaluation/configuration, not hidden labels in an observation record.

## Parameters and forward calls

`ParameterSpace` maps bounded physical quantities affinely to dimensionless
`p in [-1,1]^K`; its scales are half bound widths. Units for damage, center/width,
E0, translations, rotation, load factor and fatigue A are checked by the adapter.
Unknown names, unused inputs and search boxes beyond a damage model's legal box
are rejected. Fixed inputs are explicitly declared known quantities.

UniformDamage, GaussianDamage and LowRankDamage are continuous physical-coordinate
functions independent of Tri3. Gaussian bounds conservatively guarantee legal
damage everywhere, including between integration points. Positive Gaussian basis
functions with signed bounded coefficients have a global worst-case bound.
Candidate fields are rejected if illegal; no clipping/projection hides derivative
discontinuities. Evaluate independently on each mesh's quadrature coordinates.

`StaticLoadCase` contains a zero-damage public FE pattern, a DICMeasurement,
optional unit-bearing scalar ReactionObservation and explicit likelihood floor.
`StaticForwardAdapter` creates fresh candidate fields, solves FE, maps to DIC,
and applies declared nuisance parameters. Only identity material-state mapping is
supported. Public patterns containing hidden nonzero damage/history are rejected.
When E0 is unknown, retained pattern E0 is replaced by the search-box midpoint;
candidate E0 overwrites it at every solve. It cannot convey true unknown E0.
The current E0 nuisance is one common modulus across regions; independent region
moduli and unknown Poisson ratio are outside this implementation.

Each reaction observation has its own N units and positive standard deviation N.
Its residual is whitened separately before concatenation with DIC residuals.
No mm and N residuals are directly combined. Static x/y/shear diagnostic patterns
are independent frozen-state experiments, not general nonproportional fatigue paths.

## Finite differences and local identifiability

`centered_jacobian` evaluates legal symmetric perturbations at h and h/2, optionally
refining up to three times. Strict boundary values reject centered differentiation;
there is no clipped or one-sided fallback in the scientific diagnostic. It stores
both Jacobians, steps, column changes and a conservative roundoff-resolution
estimate `64 eps ||y|| sqrt(K) / min(h/2)`. This is a numerical heuristic rather
than a rigorous FE error bound. Both raw-mm and whitened Jacobians are saved.

`fisher_diagnostics` forms `F=Jw^T Jw` and decomposes Jw by SVD. Rank threshold is
`max(1e-6*smax, 5*FD_error_resolution, 1e-9)` in the default scientific configuration.
Null and weakest right directions are explicit. Condition is infinite when rank
deficient (JSON encodes it as null with rank/null directions retained). The
covariance object is the identifiable-subspace pseudoinverse assembled by SVD,
not finite uncertainty along null directions. Full marginal standard deviations
are reported only at full rank. Correlation heatmaps are normalized sensitivity
column products, not posterior correlation estimates.

Numerical rank is distinct from useful signal-to-noise ratio. The reported count
of singular values >1 indicates directions with local dimensionless standard
deviation <1 under the assumed noise, not a universal identifiability threshold.
All results are local; nonlinear global uniqueness is not asserted.

## Optimization, failure and history

TRF least_squares supports bounds, multi-start, explicit Gaussian prior residuals
and optional linear/Huber/soft-L1 losses. SciPy uses three-point numerical finite
differences; independent two-scale centered checks are performed at interior
converged candidates. An active bound is recorded as not checked, not mistaken
for a validated centered derivative. Scientific Fisher diagnostics are computed
at interior controlled states independently of the optimizer.

History records residual evaluations, not just accepted trust-region iterations.
All starts retain cost, status, evaluation count, optimality, message, parameters
and history, including solver diagnostics on failure. The best converged start is
chosen by the observation/prior objective. It is not selected by truth error and
does not prove a global minimum. Failed starts are never silently discarded from
the results. Priors add assumptions; prior-stabilized estimates do not increase
the rank of the DIC information alone.

## Isolation and time split

`DICMeasurement.save/load` only accepts the strict observation-only NPZ schema:
coordinates, displacement, validity, ROI, confidence and covariance. Unexpected
channels are rejected. Its acquisition metadata keeps the Phase 0 allowlist.
`evaluation.inverse_metrics.SyntheticTruth` separately stores damage, true
parameters, exact nodal displacement and internal FE fields. The inverse package
does not import evaluation/truth modules. Tests check schemas, import graph,
zero-damage public patterns and removal of unknown E0 from retained metadata.
This is an application boundary, not an OS-level security sandbox.

`FatigueForwardAdapter` supports initial damage parameters and synthetic fatigue A
only, known E0/BC/other-law inputs, ordered physical cycle counts and one proportional
load pattern. Each prediction constructs fresh MaterialHistoryState and executes
the actual Phase 1B solver at declared cycle increments. Failures cannot modify
public patterns or previous predictions. Calibration cases N≤20 are passed to
the optimizer; N=30,40 cases are held in a separate evaluation adapter. Current
state inversion, initial/law calibration and future evaluation have distinct meanings.

## Storage and reproduction

The Phase 1C CLI refuses existing output directories, retains failures and saves
configuration, full source hashes, HEAD, Python/NumPy/SciPy versions, mesh/layout/load
references, seeds, units, error tables, diagnostics and PNG/SVG figures with NPZ
numeric data. This is not true image correlation or experimentally calibrated DIC.
Local fatigue damage remains mesh sensitive; continuous parameterization prevents
index-matching mistakes but does not establish mesh-objective localization.
