"""Public API orchestration: ``roc_band``, ``roc_band_ovr``, ``from_estimator``.

Thin wiring over the validated statistical core (:mod:`rocci.band`) and the
bootstrap backend (:mod:`rocci.backend`). This layer owns ingestion, argument
validation, the warning taxonomy, and result assembly — no new statistics.
"""

from __future__ import annotations

import os
import platform
import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

import rocci.backend as _backend
from rocci._exceptions import RocciError
from rocci._result import RocBand, ScoreDiagnostics
from rocci._validation import (
    check_confidence,
    check_grid_size,
    check_n_boot,
    check_n_threads,
    resolve_seed,
)
from rocci._warnings import NormalityWarning
from rocci.band.envelope import (
    assemble_envelope_band,
    bootstrap_auc_ci,
    mann_whitney_auc,
)
from rocci.band.grids import (
    empirical_roc_on_grid,
    empirical_roc_vertices,
    grid_k_indices,
    make_grid,
)
from rocci.band.normal import normality_report, working_hotelling_band
from rocci.ingest import heavy_ties, ingest

__all__ = ["from_estimator", "roc_band", "roc_band_ovr", "show_versions"]


def roc_band(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    normal: bool = False,
    grid_size: int | None = None,
    pos_label: int | str | bool | None = None,
    score_reduce: Literal["mean", "median"] | None = None,
    nan_policy: Literal["raise", "omit"] = "raise",
    random_state: int | None = None,
    diagnostics: bool = False,
    n_threads: int | None = None,
) -> RocBand:
    """Simultaneous confidence band for a population ROC curve.

    The default (``normal=False``) is the distribution-free studentized
    bootstrap envelope with the Wilson rectangle and Beta order-statistic
    floors — calibrated across score distributions and invariant to any
    strictly monotone transform of the scores.

    Args:
        y_true: Labels in any container :mod:`rocci.ingest` can coerce.
        y_score: Scores (higher = more positive); 1-D, an ``(n, 2)``
            probability matrix, or posterior draws with ``score_reduce`` set.
        confidence: Simultaneous coverage target in ``(0, 1)``.
        n_boot: Bootstrap replicates (``>= 100``); ignored when ``normal=True``.
        normal: ``False`` → distribution-free envelope method. ``True`` →
            Working-Hotelling binormal band plus normality diagnostics.
        grid_size: FPR grid points ``K``; ``None`` → ``min(512, n_neg + 1)``.
        pos_label: Which label is positive; ``None`` infers.
        score_reduce: ``"mean"``/``"median"`` for posterior-draw scores.
        nan_policy: ``"raise"`` (default) or ``"omit"``.
        random_state: Seeds the bootstrap; same seed + backend + version ⇒
            bit-identical band. Ignored when ``normal=True``.
        diagnostics: Render the diagnostics figure immediately (a notebook
            convenience; requires matplotlib). The same figure is available
            later via :meth:`~rocci._result.RocBand.plot_diagnostics`, and
            the attribution data is always stored on the result.
        n_threads: Rust thread count; ``None`` or ``-1`` → all cores.

    Returns:
        A :class:`~rocci._result.RocBand`.

    Raises:
        RocciError: On invalid inputs (see :mod:`rocci.ingest`, ``confidence``,
            ``n_boot``), or for ``diagnostics=True`` without matplotlib.

    Examples:
        >>> import numpy as np
        >>> from rocci import roc_band
        >>> rng = np.random.default_rng(0)
        >>> y_true = np.r_[np.zeros(80), np.ones(80)]
        >>> y_score = np.r_[rng.normal(0, 1, 80), rng.normal(1.5, 1, 80)]
        >>> band = roc_band(y_true, y_score, random_state=0)
        >>> band.method
        'envelope'
        >>> bool(0.0 <= band.auc <= 1.0)
        True
    """
    alpha = check_confidence(confidence)
    if not normal:
        check_n_boot(n_boot)
    check_grid_size(grid_size)
    kernel_threads = check_n_threads(n_threads)
    data = ingest(
        y_true,
        y_score,
        pos_label=pos_label,
        score_reduce=score_reduce,
        nan_policy=nan_policy,
    )

    neg = np.sort(data.neg)
    pos = np.sort(data.pos)
    grid = make_grid(data.n_neg, grid_size)
    auc = mann_whitney_auc(neg, pos)

    if normal:
        lower, upper = working_hotelling_band(neg, pos, grid, alpha)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(
            neg, pos, fpr_v, tpr_v, heavy_ties=heavy_ties(neg, pos)
        )
        if report.suspect:
            warnings.warn(report.warning, NormalityWarning, stacklevel=2)
        result = RocBand(
            fpr=grid,
            tpr=empirical_roc_on_grid(neg, pos, grid),
            lower=lower,
            upper=upper,
            confidence=confidence,
            method="working_hotelling",
            n_neg=data.n_neg,
            n_pos=data.n_pos,
            n_boot=None,
            auc=auc,
            auc_ci=None,
            attribution=np.zeros(len(grid), dtype=np.int8),
            vacuous_below=None,
            normality=report,
            backend=_backend.BACKEND,
            random_state=None,
            notes=data.notes,
            _diag=ScoreDiagnostics(neg_sorted=neg, pos_sorted=pos),
        )
    else:
        k_indices = grid_k_indices(grid, data.n_neg)
        seed = resolve_seed(random_state)
        boot_tpr = _backend.bootstrap_tpr_matrix(
            neg, pos, k_indices, n_boot, seed, kernel_threads
        )
        band = assemble_envelope_band(boot_tpr, grid, neg, pos, alpha)
        auc_ci = bootstrap_auc_ci(boot_tpr, grid, neg, pos, alpha)
        result = RocBand(
            fpr=band.grid,
            tpr=band.tpr,
            lower=band.lower,
            upper=band.upper,
            confidence=confidence,
            method="envelope",
            n_neg=data.n_neg,
            n_pos=data.n_pos,
            n_boot=n_boot,
            auc=auc,
            auc_ci=auc_ci,
            attribution=band.attribution,
            vacuous_below=band.vacuous_below,
            normality=None,
            backend=_backend.BACKEND,
            random_state=random_state,
            notes=data.notes,
            _diag=band,
        )

    if diagnostics:
        result.plot_diagnostics()
    return result


def roc_band_ovr(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    confidence: float = 0.95,
    family: Literal["bonferroni", "none"] = "bonferroni",
    classes: Any = None,
    **roc_band_kwargs: Any,
) -> dict[Any, RocBand]:
    """One-vs-rest confidence bands for a multiclass problem.

    A thin loop over :func:`roc_band` — no new statistics beyond the alpha
    split. Column ``j`` of ``y_score`` scores ``classes[j]`` against the rest.

    Args:
        y_true: Labels with ``m > 2`` distinct classes.
        y_score: ``(n, m)`` score/probability matrix.
        confidence: Family coverage target.
        family: ``"bonferroni"`` (default) → each band at ``1 - alpha/m`` so
            joint coverage of all ``m`` curves is ``>= confidence`` (exact,
            conservative, no independence needed). ``"none"`` → each band at
            ``confidence`` marginally, no joint claim.
        classes: Class order for the columns; defaults to ``np.unique(y_true)``.
        **roc_band_kwargs: Forwarded to :func:`roc_band`.

    Returns:
        ``dict`` mapping each class label to its :class:`~rocci._result.RocBand`
        (each carrying its per-class effective ``confidence``).

    Raises:
        RocciError: If ``normal=True`` (the rest class is a mixture — the
            Working-Hotelling failure mode), if ``m <= 2``, on duplicate or
            absent class labels, on a column/class count mismatch, or on an
            invalid ``family``.

    Examples:
        >>> import numpy as np
        >>> from rocci import roc_band_ovr
        >>> rng = np.random.default_rng(0)
        >>> y = np.repeat([0, 1, 2], 40)
        >>> scores = rng.random((120, 3))
        >>> bands = roc_band_ovr(y, scores, random_state=0)
        >>> sorted(int(c) for c in bands)
        [0, 1, 2]
        >>> round(bands[0].confidence, 6)
        0.983333
    """
    if roc_band_kwargs.get("normal", False):
        raise RocciError(
            "roc_band_ovr does not support normal=True: each one-vs-rest 'rest' "
            "class is a mixture of the remaining classes — structurally the "
            "bimodal-negatives regime where Working-Hotelling coverage collapses. "
            "If you insist, loop roc_band(normal=True) per class yourself."
        )
    if family not in ("bonferroni", "none"):
        raise RocciError(f"family must be 'bonferroni' or 'none', got {family!r}.")

    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    class_list = list(np.unique(yt) if classes is None else classes)
    m = len(class_list)
    if m <= 2:
        raise RocciError(
            f"roc_band_ovr needs m > 2 classes, found {m}; for a binary problem "
            "use roc_band directly."
        )
    if len({str(c) for c in class_list}) != m:
        raise RocciError(
            f"classes contains duplicates: {class_list!r}; each class must "
            "appear exactly once."
        )
    present = np.unique(yt)
    missing = [c for c in class_list if not np.any(present == c)]
    if missing:
        raise RocciError(
            f"classes {missing!r} do not occur in y_true (present: "
            f"{present.tolist()!r}); every one-vs-rest split needs at least one "
            "positive sample."
        )
    if ys.ndim != 2 or ys.shape[1] != m:
        raise RocciError(
            f"y_score must be an (n, m={m}) matrix to match the {m} classes; got "
            f"shape {ys.shape}. Column j scores classes[j] against the rest."
        )

    alpha = 1.0 - confidence
    per_class_conf = 1.0 - alpha / m if family == "bonferroni" else confidence
    random_state = roc_band_kwargs.pop("random_state", None)
    seeds = np.random.SeedSequence(random_state).spawn(m)

    bands: dict[Any, RocBand] = {}
    for j, cls in enumerate(class_list):
        seed_j = int(seeds[j].generate_state(1, dtype=np.uint64)[0])
        bands[cls] = roc_band(
            yt == cls,
            ys[:, j],
            confidence=per_class_conf,
            random_state=seed_j,
            **roc_band_kwargs,
        )
    return bands


def from_estimator(
    estimator: Any,
    X: ArrayLike,  # noqa: N803 — sklearn convention (feature matrix)
    y: ArrayLike,
    *,
    response_method: Literal["auto", "predict_proba", "decision_function"] = "auto",
    **roc_band_kwargs: Any,
) -> RocBand:
    """Build a band from a fitted estimator's scores.

    Duck-typed like ``RocCurveDisplay.from_estimator`` — no sklearn import.
    Uses ``predict_proba(X)`` when available (the positive column is selected
    during ingestion), otherwise ``decision_function(X)``.

    Args:
        estimator: Anything exposing ``predict_proba`` and/or ``decision_function``.
        X: Feature matrix passed to the estimator.
        y: True labels.
        response_method: ``"auto"`` (default), ``"predict_proba"``, or
            ``"decision_function"``.
        **roc_band_kwargs: Forwarded to :func:`roc_band`.

    Returns:
        A :class:`~rocci._result.RocBand`.

    Raises:
        RocciError: If the requested response method is unavailable.

    Examples:
        >>> import numpy as np
        >>> from rocci import from_estimator
        >>> class Stub:
        ...     def decision_function(self, X):
        ...         return np.asarray(X).ravel()
        >>> rng = np.random.default_rng(0)
        >>> X = np.r_[rng.normal(0, 1, 60), rng.normal(1.5, 1, 60)][:, None]
        >>> y = np.r_[np.zeros(60), np.ones(60)]
        >>> band = from_estimator(Stub(), X, y, random_state=0)
        >>> band.method
        'envelope'
    """
    if response_method not in ("auto", "predict_proba", "decision_function"):
        raise RocciError(
            f"response_method must be 'auto', 'predict_proba', or "
            f"'decision_function', got {response_method!r}."
        )
    has_proba = hasattr(estimator, "predict_proba")
    has_decision = hasattr(estimator, "decision_function")

    if response_method in ("auto", "predict_proba") and has_proba:
        scores = estimator.predict_proba(X)
    elif response_method in ("auto", "decision_function") and has_decision:
        scores = estimator.decision_function(X)
    else:
        wanted = "predict_proba or decision_function"
        if response_method != "auto":
            wanted = response_method
        raise RocciError(
            f"estimator does not expose {wanted}; from_estimator needs one of "
            "predict_proba / decision_function."
        )
    return roc_band(y, scores, **roc_band_kwargs)


def show_versions() -> None:
    """Print an environment report for bug reports.

    Reports the rocci version, active backend, core dependency versions, OS,
    and CPU count. matplotlib is optional and shown as ``not installed`` when
    absent.

    Examples:
        >>> from rocci import show_versions
        >>> show_versions()  # doctest: +ELLIPSIS
        rocci...
    """
    import scipy

    from rocci import __version__

    try:
        import matplotlib

        mpl_version = matplotlib.__version__
    except ImportError:
        mpl_version = "not installed"

    lines = [
        f"rocci: {__version__}",
        f"backend: {_backend.BACKEND}",
        f"numpy: {np.__version__}",
        f"scipy: {scipy.__version__}",
        f"matplotlib: {mpl_version}",
        f"platform: {platform.platform()}",
        f"cpu_count: {os.cpu_count()}",
    ]
    print("\n".join(lines))
