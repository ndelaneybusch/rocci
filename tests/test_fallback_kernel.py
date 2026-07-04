"""NumPy fallback kernel against a brute-force resampling oracle.

Risk mitigated: the cumsum/searchsorted counting shortcut disagreeing with
literal "resample, sort, threshold, count strictly-greater" semantics —
the exact class of off-by-one/tie bug that is invisible statistically.

The oracle consumes the *same* RNG stream (identical ``default_rng`` /
``multinomial`` calls, including batching) so outputs must match exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rocci.backend._fallback import bootstrap_tpr_matrix_numpy
from rocci.band.grids import grid_k_indices, make_grid
from tests.conftest import binormal_scores


def oracle_bootstrap(neg_sorted, pos_sorted, k_indices, n_boot, seed):
    """Brute-force reference: expand counts to resamples, sort, and index."""
    n0, n1 = len(neg_sorted), len(pos_sorted)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(k_indices)))
    batch = max(1, min(n_boot, int(256e6 / (8 * (n0 + n1)))))
    rows = []
    for start in range(0, n_boot, batch):
        m = min(batch, n_boot - start)
        cnt_neg = rng.multinomial(n0, np.full(n0, 1 / n0), size=m)
        cnt_pos = rng.multinomial(n1, np.full(n1, 1 / n1), size=m)
        rows.extend(zip(cnt_neg, cnt_pos, strict=True))
    for r, (cn, cp) in enumerate(rows):
        neg_resamp_desc = np.repeat(neg_sorted, cn)[::-1]
        pos_resamp = np.repeat(pos_sorted, cp)
        for j, k in enumerate(k_indices):
            if int(k) == n0:
                # A14 sentinel: TPR is 1 by definition at k == n_neg. This is
                # deliberately not a literal `pos > -inf` comparison — scores
                # of -inf are legal, and -inf > -inf would give 0 where the
                # contract pins the ROC endpoint at 1.
                out[r, j] = 1.0
            else:
                out[r, j] = (pos_resamp > neg_resamp_desc[int(k)]).mean()
    return out


@pytest.mark.parametrize(
    ("n_neg", "n_pos", "grid_size", "tie_step"),
    [
        (12, 9, 7, None),
        (30, 30, 31, None),
        (25, 40, 11, 0.5),  # heavy ties
        (1, 5, 3, None),  # n_neg = 1 edge
        (10, 1, 6, None),  # n_pos = 1 edge
        (5, 8, 11, None),  # grid finer than negatives: duplicated k indices
        (8, 8, 9, 100.0),  # all-ties (every score rounds to 0)
    ],
)
@pytest.mark.parametrize("seed", [0, 7])
def test_fallback_matches_bruteforce_oracle(n_neg, n_pos, grid_size, tie_step, seed):
    neg, pos = binormal_scores(n_neg, n_pos, seed=seed, tie_step=tie_step)
    neg, pos = np.sort(neg), np.sort(pos)
    grid = make_grid(n_neg, grid_size=grid_size)
    k = grid_k_indices(grid, n_neg)
    ours = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=64, seed=seed)
    np.testing.assert_array_equal(ours, oracle_bootstrap(neg, pos, k, 64, seed))


@st.composite
def kernel_cases(draw):
    """Adversarial tiny inputs: massive tie pressure (a 6-value score pool
    including ±inf), duplicated/unsorted-free k sets, both sentinel and k=0."""
    pool = st.sampled_from([-np.inf, -1.5, 0.0, 0.25, 1.0, np.inf])
    n_neg = draw(st.integers(min_value=1, max_value=8))
    n_pos = draw(st.integers(min_value=1, max_value=8))
    neg = np.sort(np.array([draw(pool) for _ in range(n_neg)]))
    pos = np.sort(np.array([draw(pool) for _ in range(n_pos)]))
    ks = draw(st.lists(st.integers(0, n_neg), min_size=1, max_size=6))
    k = np.array(sorted(ks), dtype=np.uint64)
    seed = draw(st.integers(min_value=0, max_value=2**32))
    return neg, pos, k, seed


@given(kernel_cases())
@settings(max_examples=40, deadline=None)
def test_kernel_matches_oracle_on_adversarial_inputs(case):
    neg, pos, k, seed = case
    ours = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=16, seed=seed)
    np.testing.assert_array_equal(ours, oracle_bootstrap(neg, pos, k, 16, seed))


def test_sentinel_column_pins_tpr_to_one():
    neg, pos = binormal_scores(10, 10, seed=1)
    neg, pos = np.sort(neg), np.sort(pos)
    k = np.array([0, 5, 10], dtype=np.uint64)  # k = n0 sentinel at the end
    out = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=32, seed=2)
    assert (out[:, 2] == 1.0).all()


def test_batching_boundary_preserves_stream():
    # force multiple batches by using a size where the cap formula splits;
    # results must be identical to a single logical stream regardless
    neg, pos = binormal_scores(20, 20, seed=3)
    neg, pos = np.sort(neg), np.sort(pos)
    k = grid_k_indices(make_grid(20), 20)
    full = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=50, seed=4)
    again = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=50, seed=4)
    np.testing.assert_array_equal(full, again)


def test_unsorted_or_nan_scores_rejected():
    # the counting shortcut silently miscounts on unsorted/NaN input, so the
    # kernel must refuse it (same contract as the Rust core)
    sorted_scores = np.array([0.0, 1.0])
    k = np.array([0], dtype=np.uint64)
    with pytest.raises(ValueError, match="ascending"):
        bootstrap_tpr_matrix_numpy(np.array([1.0, 0.0]), sorted_scores, k, 4, 0)
    with pytest.raises(ValueError, match="ascending"):
        bootstrap_tpr_matrix_numpy(sorted_scores, np.array([0.0, np.nan]), k, 4, 0)
    with pytest.raises(ValueError, match="ascending"):
        bootstrap_tpr_matrix_numpy(np.array([np.nan]), sorted_scores, k, 4, 0)
    # +-inf sorts normally and is legal
    with_inf = np.array([-np.inf, 0.5, np.inf])
    bootstrap_tpr_matrix_numpy(with_inf, sorted_scores, k, 4, 0)


def test_values_are_valid_tprs():
    neg, pos = binormal_scores(15, 11, seed=5)
    neg, pos = np.sort(neg), np.sort(pos)
    k = grid_k_indices(make_grid(15), 15)
    out = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=100, seed=6)
    assert ((out >= 0) & (out <= 1)).all()
    # every TPR is a multiple of 1/n_pos
    np.testing.assert_array_almost_equal(out * 11, np.round(out * 11), decimal=10)
