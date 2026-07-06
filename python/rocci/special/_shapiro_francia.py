"""Shapiro-Francia W' normality test.

W' is the squared Pearson correlation between the sorted sample and the Blom
expected normal order statistics ``ndtri((i - 3/8) / (n + 1/4))`` — literally
"how straight is the normal QQ plot". Royston (1993, Statistics and
Computing 3, 175-190) gives a normal approximation for ``log(1 - W')`` whose
mean and standard deviation are simple functions of ``log(n)``, validated for
``5 <= n <= 5000``; that window is enforced here. Power is close to
Shapiro-Wilk overall and slightly better against heavy-tailed alternatives;
its weak spot (short-tailed/platykurtic departures) is covered in the band
diagnostics by the D'Agostino K² kurtosis component.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from rocci.special._normal import ndtr, ndtri

FloatArray = NDArray[np.float64]

#: Royston's p-value approximation is validated for 5 <= n <= 5000.
_MIN_N, _MAX_N = 5, 5000


def shapiro_francia(x: FloatArray) -> tuple[float, float]:
    """Shapiro-Francia normality test, one-sided against low W'.

    Args:
        x: Sample of :data:`_MIN_N` to :data:`_MAX_N` non-constant
            observations.

    Returns:
        Tuple ``(statistic, pvalue)`` — the W' statistic in ``(0, 1]`` and
        Royston's upper-tail p-value.

    Raises:
        ValueError: If the sample size is outside ``[5, 5000]`` or the sample
            is constant.

    Examples:
        >>> import numpy as np
        >>> from rocci.special import shapiro_francia
        >>> rng = np.random.default_rng(0)
        >>> stat, p = shapiro_francia(rng.normal(size=200))
        >>> bool(stat > 0.99 and p > 0.05)  # normal data: nearly straight QQ
        True
        >>> _, p = shapiro_francia(rng.lognormal(size=200))
        >>> bool(p < 1e-6)  # skewed data: decisive rejection
        True
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if not _MIN_N <= n <= _MAX_N:
        raise ValueError(
            f"shapiro_francia requires {_MIN_N} <= n <= {_MAX_N} "
            f"(Royston's validated domain), got {n}"
        )
    if np.ptp(x) == 0.0:
        raise ValueError("shapiro_francia requires a non-constant sample")

    xs = np.sort(x)
    blom = np.asarray((np.arange(1, n + 1) - 0.375) / (n + 0.25), dtype=np.float64)
    m = ndtri(blom)  # Blom expected normal order statistics
    xc = xs - xs.mean()
    mc = m - m.mean()
    denom = float(xc @ xc) * float(mc @ mc)
    if denom == 0.0:  # sample variance underflowed: numerically constant
        raise ValueError("shapiro_francia requires a non-constant sample")
    w = float((xc @ mc) ** 2) / denom
    # An affine image of the Blom scores gives W' = 1 exactly (up to rounding,
    # which can push the ratio a hair past 1); the QQ plot is perfectly
    # straight and the test carries no evidence against normality.
    if w >= 1.0:
        return 1.0, 1.0

    u = math.log(n)
    v = math.log(u)
    mu = -1.2725 + 1.0521 * (v - u)
    sigma = 1.0308 - 0.26758 * (v + 2.0 / u)
    z = (math.log(1.0 - w) - mu) / sigma
    return w, 1.0 - ndtr(z)
