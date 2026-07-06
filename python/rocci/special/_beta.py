"""Beta quantiles at positive integer shape parameters.

The band core only ever evaluates Beta quantiles of order-statistic
distributions — ``Beta(j, n + 1 - j)`` for the j-th order statistic of ``n``
uniforms — so both shapes are always positive integers. At integer shapes the
Beta CDF is a finite binomial tail::

    I_x(a, b) = P(Bin(a + b - 1, x) >= a)

which we evaluate in log space by summing whichever binomial tail is shorter:
``1 - P(Bin(n, x) <= a - 1)`` (``a`` terms) when ``a <= b``, else the
symmetric form ``P(Bin(n, 1 - x) <= b - 1)`` (``b`` terms, from
``I_x(a, b) = 1 - I_{1-x}(b, a)``). The quantile inverts this CDF by
bisection run to float adjacency, so accuracy is limited only by the CDF
evaluation itself.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

IntLike = int | np.integer


def _binom_lower(m: int, n: int, x: float) -> float:
    """``P(Bin(n, x) <= m)`` for ``0 < x < 1``, summed in log space.

    The log binomial coefficient is accumulated one ratio at a time rather
    than via ``lgamma`` differences: at large ``n`` those differences cancel
    two ~``n log n``-sized values and lose about ``log(n)`` digits, while the
    running sum of ``log((n - k) / (k + 1))`` stays accurate to a few ulps
    over the ``m + 1 <= min(a, b)`` terms ever summed here.
    """
    log_x, log_1mx = math.log(x), math.log1p(-x)
    log_comb = 0.0
    total = 0.0
    for k in range(m + 1):
        total += math.exp(log_comb + k * log_x + (n - k) * log_1mx)
        log_comb += math.log((n - k) / (k + 1.0))
    return total


def _beta_cdf(x: float, a: int, b: int) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` at integer shapes."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    n = a + b - 1
    if a <= b:
        return 1.0 - _binom_lower(a - 1, n, x)
    return _binom_lower(b - 1, n, 1.0 - x)


def _beta_ppf_scalar(q: float, a: int, b: int) -> float:
    if a < 1 or b < 1:
        raise ValueError(f"beta_ppf requires positive integer shapes, got {a=}, {b=}")
    if not 0.0 <= q <= 1.0:
        return math.nan
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    # Bisection to float adjacency: the CDF is continuous and strictly
    # increasing on (0, 1), so this converges to the ppf convention
    # inf{x : CDF(x) >= q} up to CDF rounding.
    lo, hi = 0.0, 1.0
    while True:
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            return hi
        if _beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid


@overload
def beta_ppf(q: float, a: IntLike, b: IntLike) -> float: ...
@overload
def beta_ppf(
    q: float, a: NDArray[np.integer], b: NDArray[np.integer]
) -> FloatArray: ...


def beta_ppf(
    q: float, a: IntLike | NDArray[np.integer], b: IntLike | NDArray[np.integer]
) -> float | FloatArray:
    """Beta quantile at positive integer shapes, matching ``scipy``.

    Equivalent to ``scipy.stats.beta.ppf(q, a, b)`` restricted to integer
    ``a`` and ``b`` (the order-statistic shapes the band core uses). ``a``
    and ``b`` may be scalars or broadcastable integer arrays; ``q`` is a
    scalar level applied to every shape pair.

    Args:
        q: Probability level in ``[0, 1]``; values outside map to NaN.
        a: First shape parameter(s), positive integer(s).
        b: Second shape parameter(s), positive integer(s).

    Returns:
        Quantile(s): a float for scalar shapes, else a float64 array.

    Examples:
        >>> import numpy as np
        >>> from rocci.special import beta_ppf
        >>> round(beta_ppf(0.5, 1, 1), 6)  # Beta(1, 1) is uniform
        0.5
        >>> js = np.arange(1, 4)
        >>> q = beta_ppf(0.999, js, 10 + 1 - js)  # order-statistic quantiles
        >>> bool(np.all(np.diff(q) > 0))
        True
    """
    a_arr, b_arr = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    if not (
        np.issubdtype(a_arr.dtype, np.integer)
        and np.issubdtype(b_arr.dtype, np.integer)
    ):
        raise ValueError(
            f"beta_ppf requires integer shapes, got dtypes {a_arr.dtype}, {b_arr.dtype}"
        )
    out = np.array(
        [
            _beta_ppf_scalar(float(q), int(ai), int(bi))
            for ai, bi in zip(a_arr.ravel(), b_arr.ravel(), strict=True)
        ],
        dtype=np.float64,
    )
    if a_arr.ndim == 0:
        return float(out[0])
    return out.reshape(a_arr.shape)
