"""Standard normal quantile and CDF.

``ndtri`` is Wichura's algorithm AS 241 (routine PPND16): three rational
minimax approximations — one for the central region and two for the tails on
the ``sqrt(-log(p))`` scale — giving about sixteen significant figures over
the full double-precision domain. ``ndtr`` reduces to the complementary error
function, delegated to the C library via :func:`math.erfc`.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SQRT2 = math.sqrt(2.0)

# AS 241 PPND16 coefficients, ascending powers. A/B: central rational
# approximation in r = 0.180625 - q**2 for |q| = |p - 0.5| <= 0.425.
_A = (
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
)
_B = (
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
)
# C/D: intermediate tail, in r = sqrt(-log(min(p, 1-p))) - 1.6 for r <= 5.
_C = (
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
)
_D = (
    1.0,
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
)
# E/F: far tail, in r - 5 for r > 5 (p below ~1.4e-11).
_E = (
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
)
_F = (
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
)


def _horner(coef: tuple[float, ...], r: FloatArray) -> FloatArray:
    """Evaluate a polynomial with ascending coefficients at ``r``."""
    out = np.full_like(r, coef[-1])
    for c in reversed(coef[:-1]):
        out = out * r + c
    return out


def _ndtri_array(p: FloatArray) -> FloatArray:
    out = np.full_like(p, np.nan)
    out[p == 0.0] = -np.inf
    out[p == 1.0] = np.inf

    q = p - 0.5
    central = np.abs(q) <= 0.425
    if central.any():
        qc = q[central]
        r = 0.180625 - qc * qc
        out[central] = qc * _horner(_A, r) / _horner(_B, r)

    tail = ~central & (p > 0.0) & (p < 1.0)
    if tail.any():
        pt = p[tail]
        r = np.sqrt(-np.log(np.minimum(pt, 1.0 - pt)))
        val = np.empty_like(r)
        mid = r <= 5.0
        rm = r[mid] - 1.6
        val[mid] = _horner(_C, rm) / _horner(_D, rm)
        rf = r[~mid] - 5.0
        val[~mid] = _horner(_E, rf) / _horner(_F, rf)
        out[tail] = np.where(pt < 0.5, -val, val)
    return out


@overload
def ndtri(p: float) -> float: ...
@overload
def ndtri(p: FloatArray) -> FloatArray: ...


def ndtri(p: float | FloatArray) -> float | FloatArray:
    """Standard normal quantile function (inverse CDF).

    Matches ``scipy.stats.norm.ppf``: maps ``0`` to ``-inf``, ``1`` to
    ``inf``, and anything outside ``[0, 1]`` (or NaN) to NaN.

    Args:
        p: Probability or array of probabilities.

    Returns:
        Quantiles, as a float for scalar input or an array matching the
        input shape.

    Examples:
        >>> import numpy as np
        >>> from rocci.special import ndtri
        >>> round(ndtri(0.975), 6)
        1.959964
        >>> ndtri(np.array([0.0, 0.5, 1.0]))
        array([-inf,   0.,  inf])
    """
    arr = np.asarray(p, dtype=np.float64)
    out = _ndtri_array(np.atleast_1d(arr))
    if arr.ndim == 0:
        return float(out[0])
    return out.reshape(arr.shape)


@overload
def ndtr(x: float) -> float: ...
@overload
def ndtr(x: FloatArray) -> FloatArray: ...


def ndtr(x: float | FloatArray) -> float | FloatArray:
    """Standard normal CDF, matching ``scipy.stats.norm.cdf``.

    Args:
        x: Point or array of points.

    Returns:
        CDF values, as a float for scalar input or an array matching the
        input shape.

    Examples:
        >>> import numpy as np
        >>> from rocci.special import ndtr
        >>> ndtr(0.0)
        0.5
        >>> bool(ndtr(np.array([-1.96, 1.96]))[1] > 0.97)
        True
    """
    arr = np.asarray(x, dtype=np.float64)
    flat = np.array(
        [0.5 * math.erfc(-v / _SQRT2) for v in np.atleast_1d(arr).ravel()],
        dtype=np.float64,
    )
    if arr.ndim == 0:
        return float(flat[0])
    return flat.reshape(arr.shape)
