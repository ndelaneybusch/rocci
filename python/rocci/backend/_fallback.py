"""Pure-NumPy bootstrap TPR kernel (appendix A3).

Same statistical semantics as the Rust kernel (A2) with a different RNG
stream; cross-backend agreement is distributional, not bit-wise (spec
§8.4). Vectorized over batches of replicates; only two O(K log n)
``searchsorted`` calls per replicate row run in Python-level loops.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bootstrap_tpr_matrix_numpy(
    neg_sorted: FloatArray,
    pos_sorted: FloatArray,
    k_indices: NDArray[np.uint64],
    n_boot: int,
    seed: int,
) -> FloatArray:
    """Bootstrap TPR matrix via multinomial count tallies (appendix A3).

    Per replicate: resample both classes with replacement; the TPR at grid
    point ``t`` is the fraction of resampled positives strictly greater
    than the resampled negatives' order statistic at descending 0-based
    index ``k_t`` (A14), where ``k_t = n0`` denotes a -inf sentinel
    (TPR = 1).

    Args:
        neg_sorted: Negative scores, ascending.
        pos_sorted: Positive scores, ascending.
        k_indices: Ascending order-statistic indices in ``[0, n0]`` (A14).
        n_boot: Number of bootstrap replicates.
        seed: Seed for ``np.random.default_rng``.

    Returns:
        Float64 array of shape ``(n_boot, len(k_indices))``.

    Examples:
        >>> import numpy as np
        >>> from rocci.backend._fallback import bootstrap_tpr_matrix_numpy
        >>> neg = np.array([0.0, 1.0, 2.0])
        >>> pos = np.array([1.5, 2.5, 3.5])
        >>> k = np.array([0, 1, 3], dtype=np.uint64)
        >>> out = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=8, seed=0)
        >>> out.shape
        (8, 3)
        >>> bool((out[:, 2] == 1.0).all())  # k == n0 sentinel pins TPR to 1
        True
    """
    n0, n1, n_grid = len(neg_sorted), len(pos_sorted), len(k_indices)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n_grid), dtype=np.float64)

    n_resolved = int(np.searchsorted(k_indices, n0, side="left"))  # k == n0 -> sentinel
    out[:, n_resolved:] = 1.0
    ks = k_indices[:n_resolved]

    # cap batch memory at ~256 MB of int64 count matrices
    batch = max(1, min(n_boot, int(256e6 / (8 * (n0 + n1)))))
    p_neg = np.full(n0, 1.0 / n0)
    p_pos = np.full(n1, 1.0 / n1)

    for start in range(0, n_boot, batch):
        m = min(batch, n_boot - start)
        cnt_neg = rng.multinomial(n0, p_neg, size=m)  # (m, n0) counts
        cnt_pos = rng.multinomial(n1, p_pos, size=m)

        # descending cumulative counts over negatives:
        # cum[b, i] = # resampled negs among the (i+1) largest values
        cum_neg = np.cumsum(cnt_neg[:, ::-1], axis=1)
        # ascending cumulative counts over positives:
        cum_pos = np.cumsum(cnt_pos, axis=1)

        for b in range(m):
            # smallest i with cum_neg[b, i] > k  =>  threshold = (k+1)-th largest
            i = np.searchsorted(cum_neg[b], ks, side="right")
            thr = neg_sorted[::-1][i]  # descending order values
            # strictly-greater count: n1_draws - #{resampled pos <= thr}
            pos_le = np.searchsorted(pos_sorted, thr, side="right")
            n_le = np.where(pos_le > 0, cum_pos[b][pos_le - 1], 0)
            out[start + b, :n_resolved] = (n1 - n_le) / n1
    return out
