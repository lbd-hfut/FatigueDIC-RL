"""Unit-bearing DIC covariance, PSD sampling and Cholesky likelihood factors.

Flattening is point-major: [ux0, uy0, ux1, uy1, ...]. Confidence is not
interpreted as a variance. Singular (including noiseless) sampling covariances
are valid; a Gaussian likelihood requires a separately declared positive floor.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular
from fatigue_dic_rl.core.validation import ContractError, array, scalar


@dataclass(frozen=True, eq=False)
class DisplacementCovariance:
    matrix: np.ndarray
    units: str = "mm^2"

    def __post_init__(self):
        a = array(self.matrix, name="displacement covariance")
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a) or len(a) % 2
                or not np.isfinite(a).all() or self.units != "mm^2"):
            raise ContractError("covariance must be finite even square in mm^2")
        tol = 128*np.finfo(float).eps*max(float(np.max(np.abs(a))), np.finfo(float).tiny)*len(a)
        if np.max(np.abs(a-a.T)) > tol or np.linalg.eigvalsh(a)[0] < -tol:
            raise ContractError("covariance must be symmetric positive semidefinite")
        object.__setattr__(self, "matrix", a)

    @classmethod
    def diagonal(cls, standard_deviations_mm):
        s = np.asarray(standard_deviations_mm, dtype=float)
        if s.ndim != 2 or s.shape[1] != 2 or not np.isfinite(s).all() or np.any(s < 0):
            raise ContractError("standard deviations must be nonnegative [Q,2] mm")
        return cls(np.diag(s.ravel()**2))

    @classmethod
    def per_point(cls, blocks_mm2):
        b = np.asarray(blocks_mm2, dtype=float)
        if b.ndim != 3 or b.shape[1:] != (2, 2):
            raise ContractError("point covariance must be [Q,2,2] mm^2")
        a = np.zeros((2*len(b), 2*len(b)))
        for i, block in enumerate(b):
            a[2*i:2*i+2, 2*i:2*i+2] = block
        return cls(a)

    @classmethod
    def spatial(cls, coordinates_mm, component_covariance_mm2, length_mm):
        x = np.asarray(coordinates_mm, dtype=float)
        b = np.asarray(component_covariance_mm2, dtype=float)
        ell = scalar(length_mm, "correlation length mm", minimum=0)
        if ell == 0 or x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() or b.shape != (2, 2):
            raise ContractError("spatial kernel needs finite [Q,2] mm, 2x2 covariance and ell>0")
        cls.per_point(b[None])
        r2 = np.sum((x[:, None]-x[None, :])**2, axis=2)
        return cls(np.kron(np.exp(-r2/(2*ell**2)), b))

    def selected_matrix(self, point_mask):
        mask = np.asarray(point_mask)
        if mask.dtype != bool or mask.shape != (len(self.matrix)//2,) or not mask.any():
            raise ContractError("covariance selection needs a nonempty boolean [Q] mask")
        indices = np.flatnonzero(np.repeat(mask, 2))
        return self.matrix[np.ix_(indices, indices)]

    def factor(self, point_mask, *, variance_floor_mm2=0.0):
        floor = scalar(variance_floor_mm2, "explicit variance floor mm^2", minimum=0)
        a = self.selected_matrix(point_mask)+floor*np.eye(2*np.sum(point_mask))
        try:
            l = np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ContractError("likelihood covariance is singular; declare a positive variance floor") from exc
        return CovarianceFactor(l, floor)

    def sample(self, rng, *, count=1):
        if not isinstance(rng, np.random.Generator) or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ContractError("sampling requires a Generator and positive integer count")
        w, v = np.linalg.eigh(self.matrix)
        # Only roundoff-negative eigenvalues of an already validated PSD matrix.
        root = v*np.sqrt(np.maximum(w, 0))
        return (rng.standard_normal((count, len(w))) @ root.T).reshape(count, -1, 2)


@dataclass(frozen=True, eq=False)
class CovarianceFactor:
    cholesky: np.ndarray
    variance_floor_mm2: float

    def __post_init__(self):
        a = array(self.cholesky, name="Cholesky factor")
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all() or np.any(np.diag(a) <= 0) or np.any(np.triu(a, 1)):
            raise ContractError("invalid Cholesky factor")
        object.__setattr__(self, "cholesky", a)
        object.__setattr__(self, "variance_floor_mm2", scalar(self.variance_floor_mm2, "variance floor mm^2", minimum=0))

    def whiten(self, residual):
        e = np.asarray(residual, dtype=float)
        if e.shape[0] != len(self.cholesky) or not np.isfinite(e).all():
            raise ContractError("residual/factor dimensions or finiteness disagree")
        return solve_triangular(self.cholesky, e, lower=True, check_finite=True)

    @property
    def log_determinant(self):
        return float(2*np.log(np.diag(self.cholesky)).sum())

    def negative_log_likelihood(self, residual):
        z = self.whiten(residual)
        if z.ndim != 1:
            raise ContractError("likelihood needs one residual vector")
        return float(.5*(z @ z+self.log_determinant+len(z)*np.log(2*np.pi)))
