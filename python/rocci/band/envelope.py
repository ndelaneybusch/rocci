"""Studentized envelope, assembly order, attribution, and AUC.

The assembly order (envelope -> rectangle floor + monotonicity -> Beta
floor -> pinned endpoints) is load-bearing and must not be changed:
reordering the floors will change the resulting band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from rocci.band.floors import (
    beta_floor_vacuous_below,
    beta_orderstat_floor,
    rectangle_floor,
    wilson_halfwidth_sq,
)
from rocci.band.grids import empirical_roc_on_grid, grid_k_indices

FloatArray = NDArray[np.float64]

# np.trapz was removed in favor of np.trapezoid in NumPy 2.0; support both
# because the runtime floor is numpy>=1.24.
_trapezoid: Any = getattr(np, "trapezoid", getattr(np, "trapz", None))
if _trapezoid is None:  # pragma: no cover — unreachable on numpy >= 1.24
    raise ImportError("numpy provides neither trapezoid nor trapz")

#: Attribution codes for the lower band.
ATTR_BOOTSTRAP, ATTR_BETA_FLOOR, ATTR_WILSON_FLOOR, ATTR_PINNED = 0, 1, 2, 3


@dataclass(frozen=True)
class EnvelopeBand:
    """Assembled envelope band plus the intermediates diagnostics need.

    Attributes:
        grid: FPR evaluation grid, shape ``(K,)``.
        tpr: Empirical ROC at the grid.
        lower: Final lower band (floors applied, endpoints pinned).
        upper: Final upper band.
        attribution: int8 codes per grid point — 0 bootstrap envelope,
            1 Beta floor, 2 Wilson rectangle floor, 3 pinned endpoint.
        lower_env: Pre-floor lower envelope arm, kept for attribution.
        upper_env: Pre-floor upper envelope arm.
        lower_rect: Lower band after the rectangle floor.
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
    boot_tpr: FloatArray, tpr_hat: FloatArray, alpha: float, n_neg: int, n_pos: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Studentization, KS retention, and pointwise envelope.

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
        arms are the **pre-floor arms** retained for attribution, and
        ``var_raw`` is kept for the variance-ratio gate.

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
    var_raw = boot_tpr.var(axis=0, ddof=1)  # kept for variance-ratio gate
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
    """Full envelope-band assembly with attribution.

    The pipeline order is load-bearing: envelope -> rectangle floor +
    monotonicity -> Beta floor -> pinned endpoints. Do not rerun the
    monotonicity pass after the Beta floor and do not reorder the floors.

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
    lower = beta_orderstat_floor(grid, lower_rect, neg, pos, alpha)  # applied last
    lower = lower.copy()
    upper = upper.copy()
    # Pinned endpoints. At FPR=1 the last grid point maps to the k == n_neg
    # sentinel: the threshold sits below every negative, so TPR is
    # deterministically 1 with no sampling uncertainty. A floor (e.g. the
    # Wilson rectangle) can otherwise drag lower[-1] below the envelope's
    # exact 1.0, so pin it back after all floors.
    lower[0] = 0.0
    lower[-1] = 1.0
    upper[-1] = 1.0

    tol = 1e-12
    attribution = np.zeros(len(grid), dtype=np.int8)  # 0 = bootstrap envelope
    attribution[lower_rect < lower_env - tol] = ATTR_WILSON_FLOOR
    attribution[lower < lower_rect - tol] = ATTR_BETA_FLOOR  # applied last, wins
    attribution[0] = ATTR_PINNED
    attribution[-1] = ATTR_PINNED

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


def mann_whitney_auc(neg: FloatArray, pos: FloatArray) -> float:
    """Exact Mann-Whitney AUC with ties weighted 1/2.

    This is the universal empirical-AUC convention (identical to
    ``sklearn.metrics.roc_auc_score``): the fraction of (negative, positive)
    pairs the scores order correctly, counting ties as half-correct. It is
    the point estimate reported as ``RocBand.auc``.

    Documented delta (spec §5.6 / appendix A10): the recorded paper
    implementation used a trapezoid over the A1 vertex list, which equals
    ``MW - h_last / (2 * n_neg)`` for continuous scores — systematically
    below what every other library reports. rocci reports exact MW instead;
    the golden-master fixtures do not record AUC, so nothing validated
    changes.

    Args:
        neg: Negative-class scores, any order.
        pos: Positive-class scores, any order.

    Returns:
        The Mann-Whitney AUC in ``[0, 1]``.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.envelope import mann_whitney_auc
        >>> mann_whitney_auc(np.array([0.1, 0.4]), np.array([0.35, 0.8]))
        0.75
        >>> mann_whitney_auc(np.array([0.0, 1.0]), np.array([0.0, 1.0]))  # ties
        0.5
    """
    neg = np.asarray(neg, dtype=np.float64)
    pos_asc = np.sort(np.asarray(pos, dtype=np.float64))
    n0, n1 = len(neg), len(pos_asc)
    right = np.searchsorted(pos_asc, neg, side="right")
    left = np.searchsorted(pos_asc, neg, side="left")
    n_greater = (n1 - right).sum()  # pos strictly above each neg
    n_ties = (right - left).sum()
    return float((n_greater + 0.5 * n_ties) / (n0 * n1))


def kernel_grid_auc(
    neg_sorted: FloatArray, pos_sorted: FloatArray, grid: FloatArray
) -> float:
    """Plug-in grid AUC under the bootstrap kernel's own convention.

    Evaluates exactly the functional each bootstrap replicate computes —
    TPR = fraction of positives **strictly greater** than the negative order
    statistic at index ``k_t`` (A14), trapezoid-integrated over the grid —
    on the original (unresampled) data. This is the natural center of the
    bootstrap AUC distribution and anchors the recentering in
    :func:`bootstrap_auc_ci`.

    Args:
        neg_sorted: Negative-class scores, ascending.
        pos_sorted: Positive-class scores, ascending.
        grid: FPR evaluation grid.

    Returns:
        The plug-in AUC of the kernel-convention step ROC on the grid.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.envelope import kernel_grid_auc
        >>> neg, pos = np.array([0.0, 1.0]), np.array([2.0, 3.0])
        >>> kernel_grid_auc(neg, pos, np.linspace(0, 1, 5))  # separable
        1.0
    """
    n0, n1 = len(neg_sorted), len(pos_sorted)
    k = grid_k_indices(grid, n0)
    tpr = np.ones(len(grid))
    real = k < n0  # k == n0 is the -inf sentinel (TPR = 1)
    thr = neg_sorted[::-1][k[real].astype(np.intp)]
    tpr[real] = (n1 - np.searchsorted(pos_sorted, thr, side="right")) / n1
    return float(_trapezoid(tpr, grid))


def bootstrap_auc_ci(
    boot_tpr: FloatArray,
    grid: FloatArray,
    neg_sorted: FloatArray,
    pos_sorted: FloatArray,
    alpha: float,
) -> tuple[float, float]:
    """Recentered percentile bootstrap CI for the Mann-Whitney AUC.

    Per-replicate grid AUCs are trapezoid-integrated from ``boot_tpr`` and
    their ``(alpha/2, 1 - alpha/2)`` percentiles (NumPy default ``"linear"``
    method) taken, then the interval is shifted by
    ``mann_whitney_auc - kernel_grid_auc`` and clipped to ``[0, 1]``.

    The shift is load-bearing: the kernel resamples the strictly-greater
    functional (A2/A14), whose plug-in value sits below the Mann-Whitney
    point estimate — negligibly for continuous scores, but by roughly half
    the tie mass ``P(pos = neg)`` under heavy ties, where a raw percentile
    CI can exclude the reported AUC entirely. Recentering keeps the
    bootstrap's width and shape while anchoring the interval to the
    estimator actually reported. Documented as a pointwise
    (scalar-parameter) CI.

    Args:
        boot_tpr: Bootstrap TPR matrix, shape ``(B, K)``.
        grid: FPR evaluation grid.
        neg_sorted: Negative-class scores, ascending.
        pos_sorted: Positive-class scores, ascending.
        alpha: Significance level.

    Returns:
        Tuple ``(low, high)``, clipped to ``[0, 1]``.

    Examples:
        >>> import numpy as np
        >>> from rocci.backend._fallback import bootstrap_tpr_matrix_numpy
        >>> from rocci.band.envelope import bootstrap_auc_ci, mann_whitney_auc
        >>> from rocci.band.grids import grid_k_indices, make_grid
        >>> rng = np.random.default_rng(0)
        >>> neg = np.sort(rng.normal(0, 1, 80))
        >>> pos = np.sort(rng.normal(1.5, 1, 80))
        >>> grid = make_grid(80)
        >>> boot = bootstrap_tpr_matrix_numpy(
        ...     neg, pos, grid_k_indices(grid, 80), n_boot=400, seed=1
        ... )
        >>> lo, hi = bootstrap_auc_ci(boot, grid, neg, pos, alpha=0.05)
        >>> bool(lo <= mann_whitney_auc(neg, pos) <= hi)
        True
    """
    aucs = _trapezoid(boot_tpr, grid, axis=1)
    lo, hi = np.percentile(aucs, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    shift = mann_whitney_auc(neg_sorted, pos_sorted) - kernel_grid_auc(
        neg_sorted, pos_sorted, grid
    )
    return (float(np.clip(lo + shift, 0.0, 1.0)), float(np.clip(hi + shift, 0.0, 1.0)))
