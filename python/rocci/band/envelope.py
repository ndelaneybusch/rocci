"""Studentized envelope, assembly order, attribution, and AUC (A6, A9, A10).

The assembly order (envelope -> rectangle floor + monotonicity -> Beta
floor -> pinned endpoints) is load-bearing and must not be changed: the
golden-master fixtures (spec §5.7) encode exactly this sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from rocci.band.floors import (
    beta_floor_vacuous_below,
    beta_orderstat_floor,
    rectangle_floor,
    wilson_halfwidth_sq,
)
from rocci.band.grids import empirical_roc_on_grid

FloatArray = NDArray[np.float64]

# np.trapz was removed in favor of np.trapezoid in NumPy 2.0; support both
# because the runtime floor is numpy>=1.24.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201

#: Attribution codes for the lower band (spec §3.4).
ATTR_BOOTSTRAP, ATTR_BETA_FLOOR, ATTR_WILSON_FLOOR, ATTR_PINNED = 0, 1, 2, 3


@dataclass(frozen=True)
class EnvelopeBand:
    """Assembled envelope band plus the intermediates diagnostics need.

    Attributes:
        grid: FPR evaluation grid, shape ``(K,)``.
        tpr: Empirical ROC at the grid (A1).
        lower: Final lower band (floors applied, endpoints pinned).
        upper: Final upper band.
        attribution: int8 codes per grid point — 0 bootstrap envelope,
            1 Beta floor, 2 Wilson rectangle floor, 3 pinned endpoint.
        lower_env: Pre-floor lower envelope arm (A6), kept for attribution.
        upper_env: Pre-floor upper envelope arm (A6).
        lower_rect: Lower band after the rectangle floor (A7), before A8.
        var_raw: Raw bootstrap variance per grid point (ddof=1).
        wilson_var: Wilson variance floor per grid point.
        vacuous_below: FPR below which the lower band is provably vacuous.
        alpha: Significance level the band was built at.
    """

    grid: FloatArray
    tpr: FloatArray
    lower: FloatArray
    upper: FloatArray
    attribution: NDArray[np.int8]
    lower_env: FloatArray
    upper_env: FloatArray
    lower_rect: FloatArray
    var_raw: FloatArray
    wilson_var: FloatArray
    vacuous_below: float
    alpha: float


def studentized_envelope(
    boot_tpr: FloatArray,
    tpr_hat: FloatArray,
    alpha: float,
    n_neg: int,
    n_pos: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Studentization, KS retention, and pointwise envelope (appendix A6).

    Deviations from the empirical ROC are studentized by the bootstrap SD
    floored at the Wilson variance; each replicate is scored by its supremum
    absolute studentized deviation; the ``ceil((1 - alpha) * B)`` most
    typical replicates (ties at the threshold included) are retained; the
    band is their pointwise min/max clipped to [0, 1].

    Args:
        boot_tpr: Bootstrap TPR matrix, shape ``(B, K)``.
        tpr_hat: Empirical ROC at the grid, shape ``(K,)``.
        alpha: Significance level.
        n_neg: Number of negative-class samples.
        n_pos: Number of positive-class samples.

    Returns:
        Tuple ``(lower_env, upper_env, var_raw, wilson_var)``; the envelope
        arms are the **pre-floor arm** retained for attribution (A9), and
        ``var_raw`` is kept for the A7 gate.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.envelope import studentized_envelope
        >>> rng = np.random.default_rng(0)
        >>> tpr_hat = np.linspace(0, 1, 5)
        >>> boot = np.clip(tpr_hat + rng.normal(0, 0.05, (200, 5)), 0, 1)
        >>> lo, hi, var_raw, wv = studentized_envelope(boot, tpr_hat, 0.05, 50, 50)
        >>> bool((lo <= hi).all() and (lo >= 0).all() and (hi <= 1).all())
        True
    """
    n_boot = boot_tpr.shape[0]
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    var_raw = boot_tpr.var(axis=0, ddof=1)  # keep: used by A7 gate
    wilson_var = wilson_halfwidth_sq(tpr_hat, n_pos, z_alpha) / z_alpha**2
    sd = np.sqrt(np.maximum(var_raw, wilson_var))

    eps = min(1.0 / (n_neg + n_pos), 1e-6)
    dev = boot_tpr - tpr_hat  # (B, K), signed

    # studentize with a collapse guard: where sd < eps, tiny deviations are
    # numerical noise (score 0) and real deviations are scored against eps
    stud = np.empty_like(dev)
    ok = sd >= eps
    stud[:, ok] = dev[:, ok] / sd[ok]
    low = ~ok
    stud[:, low] = np.where(np.abs(dev[:, low]) < eps, 0.0, dev[:, low] / eps)

    ks = np.abs(stud).max(axis=1)  # (B,) sup statistic
    n_retain = math.ceil((1.0 - alpha) * n_boot)
    threshold = np.sort(ks)[n_retain - 1]
    retained = boot_tpr[ks <= threshold]  # ties keep extra curves
    lower_env = np.clip(retained.min(axis=0), 0.0, 1.0)
    upper_env = np.clip(retained.max(axis=0), 0.0, 1.0)
    return lower_env, upper_env, var_raw, wilson_var


def assemble_envelope_band(
    boot_tpr: FloatArray,
    grid: FloatArray,
    neg: FloatArray,
    pos: FloatArray,
    alpha: float,
) -> EnvelopeBand:
    """Full envelope-band assembly with attribution (appendix A9).

    Pipeline order — envelope (A6) -> rectangle floor + monotonicity (A7)
    -> Beta floor (A8) -> pinned endpoints — is normative; do not re-run
    the monotonicity pass after A8 and do not reorder the floors.

    Args:
        boot_tpr: Bootstrap TPR matrix, shape ``(B, K)``.
        grid: FPR evaluation grid, shape ``(K,)``.
        neg: Negative-class scores.
        pos: Positive-class scores.
        alpha: Significance level (``1 - confidence``).

    Returns:
        An :class:`EnvelopeBand` with final arms, attribution, and the
        intermediates the diagnostics plot consumes.

    Examples:
        >>> import numpy as np
        >>> from rocci.backend._fallback import bootstrap_tpr_matrix_numpy
        >>> from rocci.band.envelope import assemble_envelope_band
        >>> from rocci.band.grids import grid_k_indices, make_grid
        >>> rng = np.random.default_rng(7)
        >>> neg = np.sort(rng.normal(0, 1, 60))
        >>> pos = np.sort(rng.normal(1.2, 1, 60))
        >>> grid = make_grid(60, grid_size=33)
        >>> boot = bootstrap_tpr_matrix_numpy(
        ...     neg, pos, grid_k_indices(grid, 60), n_boot=400, seed=1
        ... )
        >>> band = assemble_envelope_band(boot, grid, neg, pos, alpha=0.05)
        >>> bool((band.lower <= band.upper).all())
        True
        >>> int(band.attribution[0])
        3
    """
    neg = np.asarray(neg, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n_neg, n_pos = len(neg), len(pos)

    tpr_hat = empirical_roc_on_grid(neg, pos, grid)
    lower_env, upper_env, var_raw, wilson_var = studentized_envelope(
        boot_tpr, tpr_hat, alpha, n_neg, n_pos
    )
    lower_rect, upper = rectangle_floor(
        lower_env,
        upper_env,
        var_raw=var_raw,
        wilson_var=wilson_var,
        neg=neg,
        pos=pos,
        grid=grid,
        alpha=alpha,
    )
    lower = beta_orderstat_floor(grid, lower_rect, neg, pos, alpha)  # A8, last
    lower = lower.copy()
    upper = upper.copy()
    lower[0] = 0.0  # pinned endpoints
    upper[-1] = 1.0

    tol = 1e-12
    attribution = np.zeros(len(grid), dtype=np.int8)  # 0 = bootstrap envelope
    attribution[lower_rect < lower_env - tol] = ATTR_WILSON_FLOOR
    attribution[lower < lower_rect - tol] = ATTR_BETA_FLOOR  # applied last, wins
    attribution[0] = ATTR_PINNED

    return EnvelopeBand(
        grid=grid,
        tpr=tpr_hat,
        lower=lower,
        upper=upper,
        attribution=attribution,
        lower_env=lower_env,
        upper_env=upper_env,
        lower_rect=lower_rect,
        var_raw=var_raw,
        wilson_var=wilson_var,
        vacuous_below=beta_floor_vacuous_below(n_neg, alpha),
        alpha=alpha,
    )


def auc_from_vertices(fpr_v: FloatArray, tpr_v: FloatArray) -> float:
    """Trapezoid AUC over the full empirical vertex list (appendix A10).

    Args:
        fpr_v: Vertex FPR coordinates from A1 (non-decreasing).
        tpr_v: Vertex TPR coordinates from A1.

    Returns:
        The empirical AUC — computed on the vertex list, not the K grid.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.envelope import auc_from_vertices
        >>> auc_from_vertices(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 1.0]))
        0.75
    """
    return float(_trapezoid(tpr_v, fpr_v))


def bootstrap_auc_ci(
    boot_tpr: FloatArray, grid: FloatArray, alpha: float
) -> tuple[float, float]:
    """Percentile bootstrap CI of per-replicate grid AUCs (appendix A10).

    Documented as a pointwise (scalar-parameter) percentile CI.

    Args:
        boot_tpr: Bootstrap TPR matrix, shape ``(B, K)``.
        grid: FPR evaluation grid.
        alpha: Significance level.

    Returns:
        Tuple ``(low, high)`` — the empirical ``(alpha/2, 1 - alpha/2)``
        percentiles (NumPy default ``"linear"`` method).

    Examples:
        >>> import numpy as np
        >>> from rocci.band.envelope import bootstrap_auc_ci
        >>> grid = np.linspace(0, 1, 3)
        >>> boot = np.array([[0.0, 0.5, 1.0], [0.0, 0.7, 1.0], [0.0, 0.6, 1.0]])
        >>> lo, hi = bootstrap_auc_ci(boot, grid, alpha=0.5)
        >>> bool(0.5 <= lo <= hi <= 0.6)
        True
    """
    aucs = _trapezoid(boot_tpr, grid, axis=1)
    lo, hi = np.percentile(aucs, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)
