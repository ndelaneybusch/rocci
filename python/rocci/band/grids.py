"""FPR grid rule, empirical ROC, and grid index mapping.

The empirical ROC uses right-continuous step interpolation with
``>=``-threshold tie semantics on both classes; every routine that
evaluates a step function at query points shares the same
``searchsorted(side="right") - 1`` lookup logic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def default_grid_size(n_neg: int) -> int:
    """Return the default number of FPR grid points, ``min(512, n_neg + 1)``.

    Args:
        n_neg: Number of negative-class samples.

    Returns:
        Grid size ``K``.

    Examples:
        >>> from rocci.band.grids import default_grid_size
        >>> default_grid_size(30)
        31
        >>> default_grid_size(100_000)
        512
    """
    return min(512, n_neg + 1)


def make_grid(n_neg: int, grid_size: int | None = None) -> FloatArray:
    """Build the FPR evaluation grid ``linspace(0, 1, K)``.

    Args:
        n_neg: Number of negative-class samples.
        grid_size: Explicit ``K``; ``None`` uses :func:`default_grid_size`.

    Returns:
        Float64 array of shape ``(K,)`` from 0 to 1 inclusive.

    Examples:
        >>> from rocci.band.grids import make_grid
        >>> make_grid(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    k = grid_size if grid_size is not None else default_grid_size(n_neg)
    return np.linspace(0.0, 1.0, k)


def empirical_roc_vertices(
    neg: FloatArray, pos: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Compute the full empirical ROC vertex list.

    Every negative score is a threshold (``>=`` semantics on both classes),
    with ``(0, 0)`` and ``(1, 1)`` prepended/appended. The vertex list is
    the input to grid interpolation, to AUC computation, and to ``RocBand``
    evaluation.

    Args:
        neg: Negative-class scores, any order.
        pos: Positive-class scores, any order.

    Returns:
        Tuple ``(fpr_v, tpr_v)`` of float64 arrays, ``fpr_v`` non-decreasing.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.grids import empirical_roc_vertices
        >>> fpr_v, tpr_v = empirical_roc_vertices(
        ...     np.array([0.1, 0.4]), np.array([0.35, 0.8])
        ... )
        >>> fpr_v.tolist()
        [0.0, 0.5, 1.0, 1.0]
        >>> tpr_v.tolist()
        [0.0, 0.5, 1.0, 1.0]
    """
    neg_asc = np.sort(np.asarray(neg, dtype=np.float64))
    pos_asc = np.sort(np.asarray(pos, dtype=np.float64))
    n0, n1 = len(neg_asc), len(pos_asc)

    thr_desc = neg_asc[::-1]  # every negative is a threshold
    # counts with >= semantics: #{x >= v} = n - searchsorted_left(x_asc, v)
    fpr_v = (n0 - np.searchsorted(neg_asc, thr_desc, side="left")) / n0
    tpr_v = (n1 - np.searchsorted(pos_asc, thr_desc, side="left")) / n1

    fpr_v = np.concatenate(([0.0], fpr_v, [1.0]))
    tpr_v = np.concatenate(([0.0], tpr_v, [1.0]))
    return fpr_v, tpr_v


def step_lookup(x_v: FloatArray, y_v: FloatArray, query: FloatArray) -> FloatArray:
    """Evaluate a right-continuous step function at query points.

    Args:
        x_v: Step x-coordinates, non-decreasing.
        y_v: Step values at each x-coordinate.
        query: Points at which to evaluate.

    Returns:
        ``y_v[clip(searchsorted(x_v, query, side="right") - 1, 0, len - 1)]``.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.grids import step_lookup
        >>> x = np.array([0.0, 0.5, 1.0])
        >>> y = np.array([0.0, 0.7, 1.0])
        >>> step_lookup(x, y, np.array([0.25, 0.5, 0.75])).tolist()
        [0.0, 0.7, 0.7]
    """
    idx = np.searchsorted(x_v, query, side="right") - 1
    return np.asarray(y_v)[np.clip(idx, 0, len(y_v) - 1)]


def empirical_roc_on_grid(
    neg: FloatArray, pos: FloatArray, grid: FloatArray
) -> FloatArray:
    """TPR of the empirical ROC evaluated at each grid FPR.

    Runs in O(n log n) time; duplicated FPR vertices resolve to the largest
    TPR at that FPR via the ``side="right"`` lookup.

    Args:
        neg: Negative-class scores, any order.
        pos: Positive-class scores, any order.
        grid: FPR evaluation points in [0, 1].

    Returns:
        Float64 TPR array with the same shape as ``grid``.

    Examples:
        >>> import numpy as np
        >>> from rocci.band.grids import empirical_roc_on_grid
        >>> neg = np.array([0.1, 0.4])
        >>> pos = np.array([0.35, 0.8])
        >>> empirical_roc_on_grid(neg, pos, np.linspace(0, 1, 5)).tolist()
        [0.0, 0.0, 0.5, 0.5, 1.0]
    """
    fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
    return step_lookup(fpr_v, tpr_v, np.asarray(grid, dtype=np.float64))


def grid_k_indices(grid: FloatArray, n_neg: int) -> NDArray[np.uint64]:
    """Map FPR grid points to negative order-statistic indices.

    ``k = n_neg`` (which occurs only at ``t = 1.0``) acts as a -inf
    sentinel yielding TPR = 1 for that point. The result is non-decreasing
    because the grid is.

    Args:
        grid: FPR grid, non-decreasing, in [0, 1].
        n_neg: Number of negative-class samples.

    Returns:
        ``uint64`` array of 0-based indices into the descending-sorted
        negatives (``k = 0`` is the largest negative).

    Examples:
        >>> import numpy as np
        >>> from rocci.band.grids import grid_k_indices
        >>> grid_k_indices(np.linspace(0, 1, 5), n_neg=4).tolist()
        [0, 1, 2, 3, 4]
    """
    return np.clip(np.floor(grid * n_neg), 0, n_neg).astype(np.uint64)
