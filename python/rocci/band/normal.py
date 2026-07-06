"""Working-Hotelling binormal band and its normality diagnostics.

The band is a parametric alternative to the distribution-free envelope: a
method-of-moments binormal fit with delta-method uncertainty and a chi-squared
simultaneous critical value, built in probit space and mapped back to ROC
coordinates. Its coverage degrades continuously as the scores depart from
binormality, so every band is accompanied by normality diagnostics that flag a
suspect fit.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from rocci._result import NormalityReport
from rocci.special import (
    chi2_ppf,
    dagostino_k2,
    ndtr,
    ndtri,
    shapiro_francia,
    skew_kurtosis,
)

FloatArray = NDArray[np.float64]

#: Shapiro-Francia window (Royston's validated p-value domain).
_SF_MIN_N, _SF_MAX_N = 5, 5000
#: Smallest class size for D'Agostino K² (its normal approximations need it).
_K2_MIN_N = 20
#: ROC interior kept for the probit-linearity check (open interval per axis).
_INTERIOR_LO, _INTERIOR_HI = 0.05, 0.95
#: Minimum interior vertices required to estimate the probit-linearity R².
_MIN_INTERIOR_VERTICES = 10
#: A class p-value below this marks the binormal fit suspect.
_SUSPECT_P = 0.10
#: A probit-linearity R² below this marks the binormal fit suspect.
_SUSPECT_R2 = 0.98
#: Floor on class standard deviations, guarding degenerate (constant) classes.
_STD_EPS = 1e-8
#: Clip applied to grid FPRs before the probit transform (avoids +-inf).
_PROBIT_CLIP = 1e-9


def working_hotelling_band(
    neg: FloatArray, pos: FloatArray, grid: FloatArray, alpha: float
) -> tuple[FloatArray, FloatArray]:
    """Working-Hotelling binormal band on the FPR grid.

    Fits a binormal ROC by matching the class means and standard deviations,
    propagates the fit uncertainty through the delta method in probit space,
    and widens by the chi-squared(df=2) simultaneous critical value before
    mapping back to TPR. Constant or near-constant classes are guarded by a
    small standard-deviation floor.

    Args:
        neg: Negative-class scores.
        pos: Positive-class scores.
        grid: FPR evaluation grid, shape ``(K,)``.
        alpha: Significance level (``1 - confidence``).

    Returns:
        Tuple ``(lower, upper)`` on the grid, with ``lower[0] = 0`` and
        ``upper[-1] = 1`` pinned.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.normal import working_hotelling_band
        >>> rng = np.random.default_rng(0)
        >>> neg, pos = rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)
        >>> lo, hi = working_hotelling_band(neg, pos, np.linspace(0, 1, 65), 0.05)
        >>> bool((lo <= hi).all() and lo[0] == 0.0 and hi[-1] == 1.0)
        True
    """
    neg = np.asarray(neg, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n0, n1 = len(neg), len(pos)

    mu0, s0 = neg.mean(), neg.std(ddof=1)
    mu1, s1 = pos.mean(), pos.std(ddof=1)
    s0 = _STD_EPS if (math.isnan(s0) or s0 < _STD_EPS) else s0
    s1 = _STD_EPS if (math.isnan(s1) or s1 < _STD_EPS) else s1

    a = (mu1 - mu0) / s1  # binormal intercept
    b = s0 / s1  # binormal slope

    var_a = 1.0 / n1 + b * b / n0 + a * a / (2.0 * n1)
    var_b = b * b / (2.0 * n0) + b * b / (2.0 * n1)
    cov_ab = a * b / (2.0 * n1)
    if not math.isfinite(var_a):
        var_a = 1.0 / n1
    if not math.isfinite(var_b):
        var_b = 1.0 / n1
    if not math.isfinite(cov_ab):
        cov_ab = 0.0

    x = ndtri(np.clip(grid, _PROBIT_CLIP, 1.0 - _PROBIT_CLIP))  # probit FPR
    var_probit = var_a + x * x * var_b + 2.0 * x * cov_ab
    bad = ~np.isfinite(var_probit) | (var_probit < 0.0)
    var_probit[bad] = 1.0 / n1
    se = np.sqrt(var_probit)

    w = math.sqrt(chi2_ppf(1.0 - alpha, df=2))  # simultaneous critical value
    center = a + b * x
    lower = ndtr(center - w * se)
    upper = ndtr(center + w * se)
    lower[0] = 0.0
    upper[-1] = 1.0
    return lower, upper


class _ClassChecks(NamedTuple):
    """Per-class normality checks; ``nan`` marks a check that did not apply."""

    sf_stat: float
    sf_pvalue: float
    k2_stat: float
    k2_pvalue: float
    skew: float
    excess_kurtosis: float


_NO_CHECKS = _ClassChecks(*([math.nan] * 6))


def _class_checks(x: FloatArray) -> _ClassChecks:
    """Run every applicable normality check on one class's scores.

    Shapiro-Francia (QQ straightness; strongest against heavy tails) runs for
    ``5 <= n <= 5000``, D'Agostino K² (skew + kurtosis; its kurtosis leg also
    catches short-tailed/bimodal shapes) for ``n >= 20`` — both where the
    windows overlap. A class too small or with constant scores cannot be
    checked at all and reports all-``nan``, which never creates suspicion
    (``nan < threshold`` is False).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3 or np.ptp(x) == 0.0:
        return _NO_CHECKS
    skew, excess_kurtosis = skew_kurtosis(x)
    sf_stat = sf_pvalue = math.nan
    if _SF_MIN_N <= n <= _SF_MAX_N:
        sf_stat, sf_pvalue = shapiro_francia(x)
    k2_stat = k2_pvalue = math.nan
    if n >= _K2_MIN_N:
        k2_stat, k2_pvalue = dagostino_k2(x)
    return _ClassChecks(sf_stat, sf_pvalue, k2_stat, k2_pvalue, skew, excess_kurtosis)


def _probit_r2(fpr_v: FloatArray, tpr_v: FloatArray) -> float:
    """Probit-linearity R² of the empirical ROC interior.

    Keeps vertices strictly inside ``(0.05, 0.95)`` on both axes, deduplicates
    identical points, and regresses ``probit(TPR)`` on ``probit(FPR)`` by OLS.
    Returns ``nan`` when fewer than :data:`_MIN_INTERIOR_VERTICES` interior
    points remain or the probit-TPR responses have no spread (R² undefined).
    """
    fpr_v = np.asarray(fpr_v, dtype=np.float64)
    tpr_v = np.asarray(tpr_v, dtype=np.float64)
    interior = (
        (fpr_v > _INTERIOR_LO)
        & (fpr_v < _INTERIOR_HI)
        & (tpr_v > _INTERIOR_LO)
        & (tpr_v < _INTERIOR_HI)
    )
    pts = np.unique(np.column_stack([fpr_v[interior], tpr_v[interior]]), axis=0)
    if len(pts) < _MIN_INTERIOR_VERTICES:
        return math.nan

    xi = ndtri(pts[:, 0])
    yi = ndtri(pts[:, 1])
    ss_tot = float(np.sum((yi - yi.mean()) ** 2))
    if ss_tot == 0.0:
        return math.nan
    slope, intercept = np.polyfit(xi, yi, 1)
    resid = yi - (slope * xi + intercept)
    return 1.0 - float(np.sum(resid**2)) / ss_tot


def _class_clause(name: str, c: _ClassChecks) -> str:
    """One class's warning clause: check p-values plus moment effect sizes."""
    return (
        f"{name}: SF p={c.sf_pvalue:.3g}, K2 p={c.k2_pvalue:.3g}, "
        f"skew={c.skew:.2g}, excess kurtosis={c.excess_kurtosis:.2g}"
    )


def _warning_text(
    neg: _ClassChecks, pos: _ClassChecks, r2: float, *, heavy_ties: bool
) -> str:
    """Compose the suspect-binormality warning text."""
    text = (
        f"binormality looks doubtful ({_class_clause('neg', neg)}; "
        f"{_class_clause('pos', pos)}; probit-linearity R^2={r2:.3g}; "
        "Gaussian classes have skew ~= 0 and excess kurtosis ~= 0): "
        "Working-Hotelling coverage degrades continuously with departures from "
        "binormality and worsens with n, so there is no safe diagnostic region. "
        "Prefer normal=False for the distribution-free envelope band."
    )
    if heavy_ties:
        text += (
            " Scores contain heavy ties, which are incompatible with the binormal "
            "model (ties have probability zero under it)."
        )
    return text


def normality_report(
    neg: FloatArray,
    pos: FloatArray,
    fpr_v: FloatArray,
    tpr_v: FloatArray,
    *,
    heavy_ties: bool,
) -> NormalityReport:
    """Assemble the normality diagnostics for a Working-Hotelling band.

    Runs every applicable per-class check (Shapiro-Francia and D'Agostino K²)
    plus the probit-linearity R² of the empirical ROC, and flags the fit
    ``suspect`` when **any one** of them trips — any check p-value below
    :data:`_SUSPECT_P` or the R² below :data:`_SUSPECT_R2`. The OR-composition
    is deliberately sensitive: Working-Hotelling pays for every departure
    from binormality, so a false alarm costs a warning while a miss costs
    coverage. The warning text is populated only when suspect.

    Args:
        neg: Negative-class scores.
        pos: Positive-class scores.
        fpr_v: Empirical ROC vertex FPRs.
        tpr_v: Empirical ROC vertex TPRs.
        heavy_ties: Whether the pooled scores are heavily tied (adds a clause
            to the warning text).

    Returns:
        A populated :class:`~rocci._result.NormalityReport`.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.grids import empirical_roc_vertices
        >>> from rocci.band.normal import normality_report
        >>> rng = np.random.default_rng(0)
        >>> neg, pos = rng.normal(0, 1, 300), rng.normal(1.5, 1, 300)
        >>> fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        >>> rep = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        >>> bool(rep.neg_sf_pvalue > 0 and rep.neg_k2_pvalue > 0)
        True
    """
    neg_checks = _class_checks(neg)
    pos_checks = _class_checks(pos)
    r2 = _probit_r2(fpr_v, tpr_v)

    # nan p-values (check not applicable) compare False and never trigger
    pvalues = (
        neg_checks.sf_pvalue,
        neg_checks.k2_pvalue,
        pos_checks.sf_pvalue,
        pos_checks.k2_pvalue,
    )
    suspect = any(p < _SUSPECT_P for p in pvalues) or (
        not math.isnan(r2) and r2 < _SUSPECT_R2
    )
    warning = (
        _warning_text(neg_checks, pos_checks, r2, heavy_ties=heavy_ties)
        if suspect
        else ""
    )

    return NormalityReport(
        neg_sf_stat=neg_checks.sf_stat,
        neg_sf_pvalue=neg_checks.sf_pvalue,
        neg_k2_stat=neg_checks.k2_stat,
        neg_k2_pvalue=neg_checks.k2_pvalue,
        neg_skew=neg_checks.skew,
        neg_excess_kurtosis=neg_checks.excess_kurtosis,
        pos_sf_stat=pos_checks.sf_stat,
        pos_sf_pvalue=pos_checks.sf_pvalue,
        pos_k2_stat=pos_checks.k2_stat,
        pos_k2_pvalue=pos_checks.k2_pvalue,
        pos_skew=pos_checks.skew,
        pos_excess_kurtosis=pos_checks.excess_kurtosis,
        probit_r2=r2,
        suspect=suspect,
        warning=warning,
    )
