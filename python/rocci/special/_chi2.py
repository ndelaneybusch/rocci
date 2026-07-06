"""Chi-squared quantile at two degrees of freedom.

The Working-Hotelling band widens by the simultaneous chi-squared(df=2)
critical value, and chi-squared with two degrees of freedom is exactly the
exponential distribution with mean 2, so the quantile has the closed form
``-2 * log(1 - q)``. Only this case is implemented; other degrees of freedom
would need a real incomplete-gamma inverse and nothing in rocci uses one.
"""

from __future__ import annotations

import math


def chi2_ppf(q: float, df: int) -> float:
    """Chi-squared quantile, matching ``scipy.stats.chi2.ppf(q, df)``.

    Args:
        q: Probability level in ``[0, 1]``; values outside map to NaN.
        df: Degrees of freedom; only ``df=2`` is supported.

    Returns:
        The quantile ``-2 * log(1 - q)``.

    Raises:
        NotImplementedError: If ``df`` is not 2.

    Examples:
        >>> from rocci.special import chi2_ppf
        >>> round(chi2_ppf(0.95, df=2), 6)
        5.991465
    """
    if df != 2:
        raise NotImplementedError(
            f"chi2_ppf implements only df=2 (exponential closed form), got {df=}"
        )
    if not 0.0 <= q <= 1.0:
        return math.nan
    if q == 1.0:
        return math.inf
    return -2.0 * math.log1p(-q)
