"""Bounded multiplicative hazard correction with the unmodified Phase 1B transition."""
from dataclasses import dataclass
import numpy as np
from fatigue_dic_rl.core.validation import ContractError, array, scalar
from fatigue_dic_rl.fatigue.models import FatigueDamageLaw, SyntheticFatigueDamageLaw, FatigueLawParameters
from .correction_basis import CorrectionBasis


class HybridDamageLaw(FatigueDamageLaw):
    def __init__(self, prior, coefficients, basis=None, *, alpha=2., reference_energy=.025, threshold=0.):
        if not isinstance(prior, FatigueDamageLaw):
            raise ContractError('hybrid law needs an explicit FatigueDamageLaw baseline')
        self.prior = prior
        self.basis = basis or CorrectionBasis()
        self.coefficients = array(coefficients)
        self.alpha = scalar(alpha, 'alpha', minimum=0)
        self.reference_energy = scalar(reference_energy, 'Yref', minimum=0)
        self.threshold = scalar(threshold, 'threshold', minimum=0)
        if self.coefficients.shape != (self.basis.dimension,) or not np.isfinite(self.coefficients).all() or np.max(np.abs(self.coefficients)) > 4 or self.alpha > 10 or self.reference_energy <= 0:
            raise ContractError('bounded finite [K] coefficients, alpha<=10 and Yref>0 required')
        self.last_diagnostics = None

    def _hazard(self, driving, cycle, state):
        raw = self.prior.hazard(driving, cycle, state)
        x = np.maximum(driving.energy_amplitude.values/self.reference_energy-self.threshold, 0.)
        phi = self.basis.evaluate(x, state.damage.values, state.layout.coordinates, cycle.nominal_stress_ratio)
        delta = phi @ self.coefficients
        log_factor = self.alpha*np.tanh(delta)
        with np.errstate(over='raise', invalid='raise'):
            factor = np.exp(log_factor)
            values = raw.values*factor
        if not np.isfinite(values).all():
            raise ContractError('nonfinite corrected hazard')
        self.last_diagnostics = {'basis': phi, 'coefficients': self.coefficients, 'factor': array(factor),
            'delta': array(delta), 'log_factor': array(log_factor), 'prior_hazard': raw,
            'corrected_hazard': state.layout.field(values, 'cycle^-1'),
            'coefficient_sensitivity_log_hazard': array(self.alpha*(1-np.tanh(delta)**2)[:, None]*phi)}
        return self.last_diagnostics['corrected_hazard']


class DirectFeatureHazardLaw(FatigueDamageLaw):
    """Ablation with no A*x**m anchor: rate=scale*(Phi*a)**2, zero at x=0.

    This is a controlled smooth nonnegative candidate, not a known fatigue law.
    """
    def __init__(self, coefficients, basis, *, scale=8e-5, reference_energy=.025):
        self.coefficients = array(coefficients)
        self.basis, self.scale, self.reference_energy = basis, scalar(scale, 'scale', minimum=0), scalar(reference_energy, 'Yref', minimum=0)
        if self.coefficients.shape != (basis.dimension,) or not np.isfinite(self.coefficients).all() or np.max(abs(self.coefficients)) > 4 or self.reference_energy <= 0:
            raise ContractError('direct hazard requires legal finite coefficients and positive Yref')

    def _hazard(self, driving, cycle, state):
        x = driving.energy_amplitude.values/self.reference_energy
        phi = self.basis.evaluate(x, state.damage.values, state.layout.coordinates, cycle.nominal_stress_ratio)
        return state.layout.field(self.scale*(phi @ self.coefficients)**2, 'cycle^-1')


@dataclass(frozen=True)
class LawSpecification:
    mode: str = 'prior'
    dimension: int = 2
    basis_kind: str = 'energy'
    alpha: float = 2.
    beta: float = 0.
    spatial_strength: float = 0.
    structural_power: int = 1
    origin_mm: tuple = (0., 0.)
    lengths_mm: tuple = (2., 1.)

    def __post_init__(self):
        if self.mode not in ('prior', 'hybrid', 'direct', 'structural_reference'):
            raise ContractError('unknown fatigue law specification')
        CorrectionBasis(self.dimension, self.basis_kind, self.origin_mm, self.lengths_mm)
        if not np.isfinite([self.alpha, self.beta, self.spatial_strength]).all() or not 0 <= self.alpha <= 10 or self.beta < 0 or abs(self.spatial_strength) > 2:
            raise ContractError('illegal law specification')
        if isinstance(self.structural_power, bool) or self.structural_power not in (1, 2):
            raise ContractError('reference structural power must be 1 or 2')
        object.__setattr__(self, 'origin_mm', tuple(self.origin_mm))
        object.__setattr__(self, 'lengths_mm', tuple(self.lengths_mm))

    def build(self, physical, material_ids):
        from .physics_residual import StructuralReferenceLaw
        prior = SyntheticFatigueDamageLaw({int(r): FatigueLawParameters(physical['A'], .025, m=physical.get('m', 1.)) for r in np.unique(material_ids)})
        if self.mode == 'prior':
            return prior
        if self.mode == 'structural_reference':
            return StructuralReferenceLaw(prior, beta=self.beta, spatial_strength=self.spatial_strength, structural_power=self.structural_power,
                origin_mm=self.origin_mm, lengths_mm=self.lengths_mm)
        basis = CorrectionBasis(self.dimension, self.basis_kind, self.origin_mm, self.lengths_mm)
        a = [physical.get(f'c{j}', 0.) for j in range(self.dimension)]
        if self.mode == 'direct':
            return DirectFeatureHazardLaw(a, basis)
        return HybridDamageLaw(prior, a, basis, alpha=self.alpha)
