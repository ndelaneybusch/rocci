"""D'Agostino K² omnibus normality test.

The K² statistic combines two approximately standard-normal transforms — of
the sample skewness (D'Agostino 1970) and of the sample kurtosis
(Anscombe & Glynn 1983) — as ``Z_skew**2 + Z_kurt**2``, which is chi-squared
with two degrees of freedom under normality. The formulas below follow
D'Agostino, Belanger & D'Agostino Jr. (1990), matching
``scipy.stats.normaltest`` term for term; the p-value uses the closed-form
chi-squared(df=2) survival function ``exp(-K2 / 2)``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Below this the kurtosis normal approximation is unreliable (the usual
#: published guidance); rocci only tests classes far larger.
_MIN_N = 20


def _skew_z(g1: float, n: int) -> float:
    """Normalizing transform of the biased sample skewness ``g1``."""
    y = g1 * math.sqrt(((n + 1.0) * (n + 3.0)) / (6.0 * (n - 2.0)))
    beta2 = (
        3.0
        * (n * n + 27.0 * n - 70.0)
        * (n + 1.0)
        * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0))
    )
    w2 = -1.0 + math.sqrt(2.0 * (beta2 - 1.0))
    delta = 1.0 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1.0))
    if y == 0.0:
        y = 1.0  # matches scipy, which maps exactly-zero skewness to y = 1
    t = y / alpha
    return delta * math.log(t + math.sqrt(t * t + 1.0))


def _kurt_z(b2: float, n: int) -> float:
    """Normalizing transform of the (non-excess) sample kurtosis ``b2``."""
    e_b2 = 3.0 * (n - 1.0) / (n + 1.0)
    var_b2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
    x = (b2 - e_b2) / math.sqrt(var_b2)
    sqrt_beta1 = (
        6.0
        * (n * n - 5.0 * n + 2.0)
        / ((n + 7.0) * (n + 9.0))
        * math.sqrt(6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0)))
    )
    a = 6.0 + 8.0 / sqrt_beta1 * (
        2.0 / sqrt_beta1 + math.sqrt(1.0 + 4.0 / (sqrt_beta1 * sqrt_beta1))
    )
    term1 = 1.0 - 2.0 / (9.0 * a)
    denom = 1.0 + x * math.sqrt(2.0 / (a - 4.0))
    if denom == 0.0:
        return math.nan
    term2 = math.copysign(((1.0 - 2.0 / a) / abs(denom)) ** (1.0 / 3.0), denom)
    return (term1 - term2) / math.sqrt(2.0 / (9.0 * a))


def dagostino_k2(x: FloatArray) -> tuple[float, float]:
    """D'Agostino K² test, matching ``scipy.stats.normaltest``.

    Args:
        x: Sample of at least :data:`_MIN_N` non-constant observations.

    Returns:
        Tuple ``(statistic, pvalue)``.

    Raises:
        ValueError: If the sample has fewer than :data:`_MIN_N` observations
            or zero variance.

    Examples:
        >>> import numpy as np
        >>> from rocci.special import dagostino_k2
        >>> rng = np.random.default_rng(0)
        >>> stat, p = dagostino_k2(rng.normal(size=6000))
        >>> bool(p > 0.05)  # normal data: no evidence against normality
        True
        >>> _, p = dagostino_k2(rng.lognormal(size=6000))
        >>> bool(p < 1e-10)  # heavily skewed data: decisive rejection
        True
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < _MIN_N:
        raise ValueError(f"dagostino_k2 requires n >= {_MIN_N}, got {n}")
    xc = x - x.mean()
    m2 = float(np.mean(xc**2))
    if m2 == 0.0:
        raise ValueError("dagostino_k2 requires a non-constant sample")
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))

    z_s = _skew_z(m3 / m2**1.5, n)
    z_k = _kurt_z(m4 / (m2 * m2), n)
    k2 = z_s * z_s + z_k * z_k
    return k2, math.exp(-k2 / 2.0)
