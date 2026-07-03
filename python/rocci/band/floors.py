"""Wilson score machinery and the exact band floors.

Three mechanisms repair the bootstrap envelope where resampling variance
collapses or goes blind:

- the Wilson variance floor used during studentization;
- the variance-ratio gated Wilson rectangle floor, which widens the band
  wherever raw bootstrap variance falls below even the binomial component;
- the Beta order-statistic floor, which *lowers* an overconfident lower
  band at extreme FPR, where the dominant uncertainty is the horizontal
  location of the threshold and no variance-based mechanism can see it.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from rocci.band.grids import empirical_roc_on_grid

FloatArray = NDArray[np.float64]

#: Number of negative order statistics used by the Beta floor.
#: This is a fixed design parameter, deliberately not exposed for tuning.
J_MAX = 25


def wilson_bounds(p: FloatArray, n: int, z: float) -> tuple[FloatArray, FloatArray]:
    """Two-sided Wilson interval, clipped to [0, 1].

    Args:
        p: Proportion estimates.
        n: Number of Bernoulli trials; ``n <= 0`` returns ``(0s, 1s)``.
        z: Normal critical value (``> 0``).

    Returns:
        Tuple ``(lower, upper)`` arrays.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.floors import wilson_bounds
        >>> lo, hi = wilson_bounds(np.array([0.0, 0.5, 1.0]), n=100, z=1.96)
        >>> bool((hi > lo).all())
        True
        >>> bool(lo[0] == 0.0 and hi[0] > 0.0)  # non-degenerate at p=0
        True
    """
    p = np.asarray(p, dtype=np.float64)
    if n <= 0:
        return np.zeros_like(p), np.ones_like(p)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)


def wilson_halfwidth_sq(p: FloatArray, n: int, z: float) -> FloatArray:
    """Squared Wilson half-width; strictly positive even at p in {0, 1}.

    Used as the studentization variance floor: divide by ``z**2`` to convert
    a half-width^2 at level ``z`` into a variance-scale quantity.

    Args:
        p: Proportion estimates.
        n: Number of Bernoulli trials; ``n <= 0`` returns zeros.
        z: Normal critical value.

    Returns:
        Array of squared half-widths.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.floors import wilson_halfwidth_sq
        >>> v = wilson_halfwidth_sq(np.array([0.0, 0.5, 1.0]), n=100, z=1.96)
        >>> bool((v > 0).all())
        True
    """
    p = np.asarray(p, dtype=np.float64)
    if n <= 0:
        return np.zeros_like(p)
    denom = 1.0 + z * z / n
    return (z * z / denom**2) * (p * (1.0 - p) / n + z * z / (4.0 * n * n))


def wilson_lower_one_sided(p: FloatArray, n: int, alpha: float) -> FloatArray:
    """One-sided Wilson lower bound at level ``alpha``.

    Args:
        p: Proportion estimates.
        n: Number of Bernoulli trials.
        alpha: One-sided significance level.

    Returns:
        Lower confidence bounds, clipped to [0, 1].

    Examples:
        >>> import numpy as np
        >>> from rocci.band.floors import wilson_lower_one_sided
        >>> lo = wilson_lower_one_sided(np.array([0.9]), n=50, alpha=0.05)
        >>> bool(0.0 < lo[0] < 0.9)
        True
    """
    p = np.asarray(p, dtype=np.float64)
    z1 = float(norm.ppf(1.0 - alpha))
    denom = 1.0 + z1 * z1 / n
    center = (p + z1 * z1 / (2.0 * n)) / denom
    half = (z1 / denom) * np.sqrt(p * (1.0 - p) / n + z1 * z1 / (4.0 * n * n))
    return np.clip(center - half, 0.0, 1.0)


def _corner_envelope(
    x: FloatArray, y: FloatArray, grid: FloatArray, take: str
) -> FloatArray:
    """Project rectangle corner points onto the grid.

    Sort by x, collapse duplicate x (min for lower / max for upper), then
    linear interpolation with edge clamping (``np.interp`` semantics).
    """
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    ux, inv = np.unique(xs, return_inverse=True)
    fill = np.inf if take == "min" else -np.inf
    uy = np.full(len(ux), fill)
    (np.minimum if take == "min" else np.maximum).at(uy, inv, ys)
    return np.interp(grid, ux, uy)


def wilson_rectangle_band(
    neg: FloatArray, pos: FloatArray, grid: FloatArray, alpha: float
) -> tuple[FloatArray, FloatArray]:
    """Pointwise Wilson 2-D rectangle band.

    Used **only** as the floor source inside the variance-ratio gate,
    always with Šidák correction across each rectangle's two margins.

    Args:
        neg: Negative-class scores.
        pos: Positive-class scores.
        grid: FPR evaluation grid.
        alpha: Significance level for each joint rectangle.

    Returns:
        Tuple ``(lower, upper)`` on the grid, with ``lower[0] = 0`` and
        ``upper[-1] = 1`` pinned.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.floors import wilson_rectangle_band
        >>> rng = np.random.default_rng(0)
        >>> neg, pos = rng.normal(0, 1, 50), rng.normal(1, 1, 50)
        >>> lo, hi = wilson_rectangle_band(neg, pos, np.linspace(0, 1, 11), 0.05)
        >>> bool((lo <= hi).all())
        True
    """
    alpha_m = 1.0 - math.sqrt(1.0 - alpha)  # Šidák across 2 margins
    z = float(norm.ppf(1.0 - alpha_m / 2.0))
    n0, n1 = len(neg), len(pos)

    grid = np.asarray(grid, dtype=np.float64)
    tpr = empirical_roc_on_grid(neg, pos, grid)
    fpr_lo, fpr_hi = wilson_bounds(grid, n0, z)  # uncertainty in FPR
    tpr_lo, tpr_hi = wilson_bounds(tpr, n1, z)  # uncertainty in TPR

    # upper envelope from upper-left corners (optimistic); lower from
    # lower-right corners (pessimistic)
    upper = _corner_envelope(x=fpr_lo, y=tpr_hi, grid=grid, take="max")
    lower = _corner_envelope(x=fpr_hi, y=tpr_lo, grid=grid, take="min")
    lower[0] = 0.0
    upper[-1] = 1.0
    return lower, upper


def rectangle_floor(
    lower_env: FloatArray,
    upper_env: FloatArray,
    *,
    var_raw: FloatArray,
    wilson_var: FloatArray,
    neg: FloatArray,
    pos: FloatArray,
    grid: FloatArray,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    """Variance-ratio gate + Wilson rectangle floor.

    Detects where the **raw** (unfloored) bootstrap variance has collapsed
    below even the binomial component and floors those points with the
    rectangle band at a Šidák-corrected level, then enforces band
    monotonicity. The floor can only widen the band.

    Args:
        lower_env: Pre-floor lower envelope arm.
        upper_env: Pre-floor upper envelope arm.
        var_raw: Raw bootstrap variance per grid point (ddof=1).
        wilson_var: Wilson variance floor per grid point.
        neg: Negative-class scores.
        pos: Positive-class scores.
        grid: FPR evaluation grid.
        alpha: Band significance level.

    Returns:
        Tuple ``(lower, upper)`` after flooring and monotonicity.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.floors import rectangle_floor, wilson_halfwidth_sq
        >>> rng = np.random.default_rng(1)
        >>> neg, pos = rng.normal(0, 1, 40), rng.normal(2, 1, 40)
        >>> grid = np.linspace(0, 1, 9)
        >>> lo_env = np.full(9, 0.6)
        >>> hi_env = np.full(9, 0.6)
        >>> wv = wilson_halfwidth_sq(np.full(9, 0.6), 40, 1.96) / 1.96**2
        >>> lo, hi = rectangle_floor(
        ...     lo_env,
        ...     hi_env,
        ...     var_raw=np.zeros(9),
        ...     wilson_var=wv,
        ...     neg=neg,
        ...     pos=pos,
        ...     grid=grid,
        ...     alpha=0.05,
        ... )
        >>> bool((lo <= lo_env).all() and (hi >= hi_env).all())  # only widens
        True
    """
    r = var_raw / np.clip(wilson_var, 1e-30, None)
    deficiency = np.clip(1.0 - r, 0.0, None)  # (K,), >0 where collapsed
    k_eff = float(deficiency.sum())
    alpha_w = 1.0 - (1.0 - alpha) ** (1.0 / k_eff) if k_eff > 1.0 else alpha

    lower, upper = lower_env.copy(), upper_env.copy()
    needs = deficiency > 0
    if needs.any():
        rect_lo, rect_hi = wilson_rectangle_band(neg, pos, grid, alpha_w)
        lower[needs] = np.minimum(lower[needs], rect_lo[needs])
        upper[needs] = np.maximum(upper[needs], rect_hi[needs])
        # enforce band monotonicity (ROC bands are non-decreasing):
        upper = np.maximum.accumulate(upper)
        lower = np.minimum.accumulate(lower[::-1])[::-1]
    return lower, upper


def beta_orderstat_floor(
    grid: FloatArray,
    lower: FloatArray,
    neg: FloatArray,
    pos: FloatArray,
    alpha: float,
    j_max: int = J_MAX,
) -> FloatArray:
    """Exact Beta order-statistic floor for the lower band.

    For continuous scores the true FPR exceedance at the j-th largest
    negative is exactly ``Beta(j, n0 + 1 - j)`` regardless of the score
    distribution. Applied as a pointwise **minimum**: inside its
    jurisdiction it lowers an overconfident envelope to the certified bound
    (an honesty repair — it never raises the band); outside, ``+inf`` makes
    it a no-op. With ties the true exceedance is stochastically smaller
    than the Beta law, so discrete scores err conservative.

    Args:
        grid: FPR evaluation grid.
        lower: Current lower band.
        neg: Negative-class scores.
        pos: Positive-class scores.
        alpha: Total alpha budget for the floor (Bonferroni over
            ``2 * j_max`` one-sided events).
        j_max: Number of order statistics used; fixed at :data:`J_MAX` in
            the public pipeline.

    Returns:
        Lower band with the floor applied.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.floors import beta_orderstat_floor
        >>> rng = np.random.default_rng(2)
        >>> neg, pos = rng.normal(0, 1, 100), rng.normal(3, 1, 100)
        >>> grid = np.linspace(0, 1, 101)
        >>> floored = beta_orderstat_floor(grid, np.ones(101), neg, pos, 0.05)
        >>> bool(floored[1] < 1.0)  # overconfident lower band gets repaired
        True
    """
    n0, n1 = len(neg), len(pos)
    j_used = min(j_max, n0)
    if j_used == 0 or n1 == 0:
        return lower
    a_e = alpha / (2 * j_max)  # Bonferroni over 2*j_max one-sided events
    js = np.arange(1, j_used + 1)
    q = beta_dist.ppf(1.0 - a_e, js, n0 + 1 - js)  # jurisdiction edges, increasing

    # empirical TPR at the j-th largest negative, strictly-greater semantics
    neg_desc = np.sort(neg)[::-1][:j_used]
    pos_asc = np.sort(pos)
    tpr_hat_j = (n1 - np.searchsorted(pos_asc, neg_desc, side="right")) / n1

    # one-sided Wilson lower bounds at level a_e; prepend 0 so that
    # "no qualifying order statistic" (j_star == 0) maps to a vacuous floor
    bounds = np.concatenate(([0.0], wilson_lower_one_sided(tpr_hat_j, n1, a_e)))

    zone = (grid > 0.0) & (grid <= q[-1])
    if not zone.any():
        return lower
    floor = np.full_like(grid, np.inf)
    j_star = np.searchsorted(q, grid[zone], side="right")  # largest j with q_j <= t
    floor[zone] = bounds[j_star]
    return np.minimum(lower, floor)


def beta_floor_vacuous_below(n_neg: int, alpha: float, j_max: int = J_MAX) -> float:
    """FPR below which the lower band is provably vacuous (``q_1``).

    Args:
        n_neg: Number of negative-class samples.
        alpha: Band significance level.
        j_max: Beta floor order-statistic budget.

    Returns:
        ``Beta(1, n_neg).ppf(1 - alpha / (2 * j_max))``.

    Examples:
        >>> from rocci.band.floors import beta_floor_vacuous_below
        >>> q1 = beta_floor_vacuous_below(n_neg=100, alpha=0.05)
        >>> bool(0.0 < q1 < 0.1)
        True
    """
    a_e = alpha / (2 * j_max)
    return float(beta_dist.ppf(1.0 - a_e, 1, n_neg))
