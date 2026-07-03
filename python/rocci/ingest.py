"""Score and label ingestion.

Accepts what users actually have — NumPy arrays, pandas/polars Series, torch or
JAX tensors, ``(n, 2)`` probability matrices, posterior-predictive draws, plain
Python sequences — with **zero hard dependencies** on any of those libraries.
All coercion is duck-typed / protocol-based: rocci never imports torch, pandas,
polars, or xarray. The output is the ``(neg, pos)`` score split that the
statistical core (:mod:`rocci.band`) consumes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocci._exceptions import RocciError
from rocci._warnings import SmallSampleWarning, TiesWarning

FloatArray = NDArray[np.float64]

#: Fraction of distinct scores below which the heavy-ties warning fires.
_TIES_FRACTION = 0.5
#: Row count sampled when sniffing an ``(n, 2)`` matrix for probabilities.
_PROBA_SNIFF_ROWS = 256


@dataclass(frozen=True)
class IngestedData:
    """Validated, split score arrays plus reporting metadata.

    Attributes:
        neg: Negative-class scores (unsorted; the API layer sorts them).
        pos: Positive-class scores (unsorted).
        n_neg: Number of negatives.
        n_pos: Number of positives.
        notes: INFO-level notes (e.g. probability-column selection) surfaced
            by :meth:`rocci.RocBand.summary`, never raised as warnings.
    """

    neg: FloatArray
    pos: FloatArray
    n_neg: int
    n_pos: int
    notes: tuple[str, ...]


def _coerce(x: object, name: str) -> np.ndarray:
    """Coerce one input container to a NumPy array (ordered protocol attempts).

    Args:
        x: The raw input (ndarray, tensor, Series, sequence, ...).
        name: Argument name, for error messages.

    Returns:
        A NumPy array view/copy of ``x``.

    Raises:
        RocciError: If ``x`` cannot be coerced by any supported protocol.

    Examples:
        >>> from rocci.ingest import _coerce
        >>> _coerce([1, 2, 3], "y_score").tolist()
        [1, 2, 3]
    """
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__dlpack__"):
        try:
            return np.from_dlpack(x)  # ty: ignore[invalid-argument-type]
        except (TypeError, RuntimeError, BufferError, ValueError):
            # e.g. a CUDA tensor: fall back to the framework's own transfer path.
            if all(hasattr(x, m) for m in ("detach", "cpu", "numpy")):
                return np.asarray(x.detach().cpu().numpy())  # ty: ignore[unresolved-attribute]
            raise RocciError(
                f"{name} implements __dlpack__ but could not be converted to a "
                "NumPy array (is it on a non-CPU device without .detach().cpu()"
                ".numpy()?). Move it to CPU / NumPy before calling rocci."
            ) from None
    if hasattr(x, "__array__"):
        return np.asarray(x)
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy())  # ty: ignore[call-non-callable]
    try:
        return np.asarray(x)
    except (TypeError, ValueError) as err:
        raise RocciError(
            f"{name} could not be interpreted as an array; pass a NumPy array, a "
            "pandas/polars Series, a tensor, or a Python sequence."
        ) from err


def _as_float(a: np.ndarray, name: str = "y_score") -> FloatArray:
    """Cast an array to contiguous float64, or raise a numeric-input error."""
    try:
        return np.ascontiguousarray(a, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise RocciError(
            f"{name} must be numeric; convert it to floating-point scores "
            "(higher = more positive) before calling rocci."
        ) from err


def _label_mapping(labels: np.ndarray, pos_label: object) -> tuple[object, int]:
    """Resolve which label value is positive and its column position.

    Operates on NaN-free labels so missing values never inflate the class
    count. Returns the positive value and its index (0 or 1) among the two
    sorted distinct labels — the latter selects the ``(n, 2)`` probability
    column.

    Args:
        labels: Coerced, NaN-free label array (any dtype).
        pos_label: Caller-supplied positive label, or ``None`` to infer.

    Returns:
        Tuple ``(pos_value, pos_col)``.

    Raises:
        RocciError: On >2 classes, a single class, an unusable ``pos_label``,
            or an un-inferable pair.
    """
    if labels.dtype == bool:
        return True, 1

    uniques = np.unique(labels)
    if len(uniques) > 2:
        raise RocciError(
            f"y_true has {len(uniques)} distinct labels; there is no single ROC "
            "curve for m > 2 classes. Use roc_band_ovr for one-vs-rest bands with "
            "a family-wise guarantee, or binarize and pass pos_label."
        )
    if len(uniques) < 2:
        raise RocciError(
            "y_true has only one class present; a ROC curve needs both a positive "
            "and a negative class."
        )

    if pos_label is not None:
        if not bool(np.any(uniques == pos_label)):
            raise RocciError(
                f"pos_label={pos_label!r} is not among the labels {uniques.tolist()!r}."
            )
        pos_value: object = pos_label
    else:
        label_set = set(uniques.tolist())
        if label_set == {0, 1} or label_set == {-1, 1}:
            pos_value = 1
        else:
            raise RocciError(
                f"cannot infer the positive class from labels {uniques.tolist()!r}; "
                "pass pos_label=<the positive label> explicitly."
            )
    pos_col = int(np.flatnonzero(uniques == pos_value)[0])
    return pos_value, pos_col


def _rows_sum_to_one(mat: FloatArray) -> bool:
    """Sniff whether a 2-column matrix looks like ``predict_proba`` output."""
    sample = mat[: min(len(mat), _PROBA_SNIFF_ROWS)]
    return bool(np.allclose(sample.sum(axis=1), 1.0, atol=1e-6))


def _reduce_scores(
    y_score: np.ndarray, score_reduce: str | None, pos_col: int
) -> tuple[FloatArray, list[str]]:
    """Reduce raw scores to a 1-D positive-class score vector.

    Args:
        y_score: Coerced score array (1-D, ``(n, 2)`` proba, or draw axes).
        score_reduce: ``"mean"`` / ``"median"`` for posterior draws, else ``None``.
        pos_col: Positive-class column for ``(n, 2)`` probabilities.

    Returns:
        Tuple ``(scores, notes)`` — the 1-D float score vector and any INFO notes.

    Raises:
        RocciError: On ambiguous multi-dimensional input without ``score_reduce``.
    """
    arr = _as_float(y_score)
    notes: list[str] = []

    if arr.ndim == 1:
        return arr, notes

    if arr.ndim == 2 and arr.shape[1] == 2 and _rows_sum_to_one(arr):
        notes.append(
            f"y_score looked like (n, 2) predict_proba output; used column "
            f"{pos_col} as the positive-class score."
        )
        return np.ascontiguousarray(arr[:, pos_col]), notes

    if arr.ndim in (2, 3):
        if score_reduce is None:
            raise RocciError(
                f"y_score has shape {arr.shape}, which looks like posterior-"
                "predictive draws (draws, n) or (chain, draw, n). Pass "
                "score_reduce='mean' or 'median' to reduce over the draw axes. "
                "(A 2-column matrix is only auto-detected as probabilities when "
                "its rows sum to 1.)"
            )
        if score_reduce not in ("mean", "median"):
            raise RocciError(
                f"score_reduce must be 'mean', 'median', or None, got {score_reduce!r}."
            )
        reducer = np.mean if score_reduce == "mean" else np.median
        axes = tuple(range(arr.ndim - 1))  # reduce every axis but the last (n)
        notes.append(
            f"reduced {arr.ndim}-D scores over the draw axes with "
            f"score_reduce={score_reduce!r}; the band quantifies sampling "
            "uncertainty of the reduced scores, not posterior uncertainty."
        )
        return np.ascontiguousarray(reducer(arr, axis=axes)), notes

    raise RocciError(
        f"y_score has unsupported shape {arr.shape}; pass 1-D scores, an (n, 2) "
        "probability matrix, or posterior draws with score_reduce set."
    )


def _apply_nan_policy(
    is_pos: NDArray[np.bool_], scores: FloatArray, nan_policy: str
) -> tuple[NDArray[np.bool_], FloatArray]:
    """Enforce ``nan_policy`` over aligned labels and scores."""
    if nan_policy not in ("raise", "omit"):
        raise RocciError(f"nan_policy must be 'raise' or 'omit', got {nan_policy!r}.")
    nan_mask = np.isnan(scores)
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return is_pos, scores
    if nan_policy == "raise":
        raise RocciError(
            f"found {n_nan} NaN value(s) across y_true/y_score. Pass "
            "nan_policy='omit' to drop those rows, or clean the inputs first. "
            "(±inf scores are allowed; only NaN is rejected.)"
        )
    warnings.warn(
        f"nan_policy='omit': dropped {n_nan} row(s) containing NaN.",
        SmallSampleWarning,
        stacklevel=3,
    )
    keep = ~nan_mask
    return is_pos[keep], scores[keep]


def heavy_ties(neg: FloatArray, pos: FloatArray) -> bool:
    """Whether the pooled scores are heavily tied.

    True when the fraction of distinct pooled scores falls below
    :data:`_TIES_FRACTION` — the band stays valid but conservative, and the
    binormal model becomes untenable (ties have probability zero under it).

    Args:
        neg: Negative-class scores.
        pos: Positive-class scores.

    Returns:
        ``True`` if the distinct-score fraction is below the threshold.

    Examples:
        >>> import numpy as np
        >>> from rocci.ingest import heavy_ties
        >>> heavy_ties(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]))
        True
        >>> heavy_ties(np.arange(10.0), np.arange(10.0, 20.0))
        False
    """
    combined = np.concatenate([neg, pos])
    return len(np.unique(combined)) / len(combined) < _TIES_FRACTION


def _check_ties(neg: FloatArray, pos: FloatArray) -> None:
    """Warn on heavy ties or a constant-score class."""
    for label, arr in (("negative", neg), ("positive", pos)):
        if len(np.unique(arr)) == 1:
            warnings.warn(
                f"the {label} class has constant scores; the band degenerates to "
                "the exact floors and will be honestly wide, but stays valid.",
                TiesWarning,
                stacklevel=3,
            )
    if heavy_ties(neg, pos):
        combined = np.concatenate([neg, pos])
        n = len(combined)
        n_unique = len(np.unique(combined))
        warnings.warn(
            f"scores are heavily tied ({n_unique} distinct of {n}); the band stays "
            "valid but conservative (the Beta floor errs safe under ties).",
            TiesWarning,
            stacklevel=3,
        )


def ingest(
    y_true: object,
    y_score: object,
    *,
    pos_label: object = None,
    score_reduce: str | None = None,
    nan_policy: str = "raise",
) -> IngestedData:
    """Coerce, validate, and split inputs into positive/negative scores.

    Args:
        y_true: Labels in any supported container (bool, ``{0,1}``,
            ``{-1,1}``, or two distinct values of any dtype).
        y_score: Scores in any supported container (higher = more
            positive); 1-D, an ``(n, 2)`` probability matrix, or posterior draws.
        pos_label: Which label is positive; ``None`` infers it from the labels.
        score_reduce: ``"mean"``/``"median"`` for posterior-draw scores.
        nan_policy: ``"raise"`` (default) or ``"omit"``.

    Returns:
        An :class:`IngestedData` with the split scores and reporting notes.

    Raises:
        RocciError: On >2 label classes, a single class, too-small classes,
            un-inferable labels, ambiguous score shapes, or NaN under
            ``nan_policy='raise'``.

    Examples:
        >>> import numpy as np
        >>> from rocci.ingest import ingest
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
        >>> data = ingest(y_true, y_score)
        >>> data.n_neg, data.n_pos
        (2, 2)
    """
    yt = _coerce(y_true, "y_true")
    label_nan = np.isnan(yt) if yt.dtype.kind == "f" else np.zeros(len(yt), bool)
    pos_value, pos_col = _label_mapping(yt[~label_nan], pos_label)

    scores, notes = _reduce_scores(_coerce(y_score, "y_score"), score_reduce, pos_col)
    if len(scores) != len(yt):
        raise RocciError(
            f"y_true has {len(yt)} rows but y_score reduced to {len(scores)}; "
            "the two inputs must describe the same samples."
        )

    is_pos = (yt == pos_value) if yt.dtype != bool else yt.astype(bool)
    # Fold NaN labels into the score-NaN mask so the policy handles both together
    # (raise, or drop the row) without silently pre-filtering.
    scores_masked = scores.astype(np.float64, copy=True)
    scores_masked[label_nan] = np.nan
    is_pos, scores = _apply_nan_policy(is_pos, scores_masked, nan_policy)

    neg = np.ascontiguousarray(scores[~is_pos])
    pos = np.ascontiguousarray(scores[is_pos])
    n_neg, n_pos = len(neg), len(pos)

    if n_neg == 0 or n_pos == 0:
        raise RocciError(
            "only one class remains after ingestion; a ROC curve needs both a "
            f"positive and a negative class (n_neg={n_neg}, n_pos={n_pos})."
        )
    if n_neg < 2 or n_pos < 2:
        raise RocciError(
            f"each class needs at least 2 samples, got n_neg={n_neg}, n_pos={n_pos}."
        )
    if n_neg < 20 or n_pos < 20:
        warnings.warn(
            f"small sample (n_neg={n_neg}, n_pos={n_pos}); the band will be "
            "dominated by the exact floors, but remains valid.",
            SmallSampleWarning,
            stacklevel=2,
        )

    _check_ties(neg, pos)
    return IngestedData(neg=neg, pos=pos, n_neg=n_neg, n_pos=n_pos, notes=tuple(notes))
