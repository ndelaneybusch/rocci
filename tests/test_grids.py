"""Grid rule, empirical ROC, step lookup, and index mapping.

Risk mitigated: the O(n log n) searchsorted formulation silently diverging
from the >=-threshold step semantics, especially under ties.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocci.band.grids import (
    default_grid_size,
    empirical_roc_on_grid,
    empirical_roc_vertices,
    grid_k_indices,
    make_grid,
    step_lookup,
)
from tests.conftest import binormal_scores


def roc_on_grid_reference(neg, pos, grid):
    """O(n^2) brute-force reference: every negative is a threshold, >= counting
    on both classes, right-continuous step lookup."""
    thr = np.sort(neg)[::-1]
    fpr_v = np.concatenate(([0.0], [(neg >= t).mean() for t in thr], [1.0]))
    tpr_v = np.concatenate(([0.0], [(pos >= t).mean() for t in thr], [1.0]))
    idx = np.searchsorted(fpr_v, grid, side="right") - 1
    return tpr_v[np.clip(idx, 0, len(tpr_v) - 1)]


class TestGridRule:
    def test_default_grid_size_small_n(self):
        assert default_grid_size(30) == 31

    def test_default_grid_size_caps_at_512(self):
        assert default_grid_size(100_000) == 512

    def test_make_grid_spans_unit_interval(self):
        grid = make_grid(100)
        assert grid[0] == 0.0
        assert grid[-1] == 1.0
        assert len(grid) == 101
        assert (np.diff(grid) > 0).all()

    def test_make_grid_explicit_size_overrides(self):
        assert len(make_grid(100, grid_size=17)) == 17


class TestEmpiricalRoc:
    @pytest.mark.parametrize(
        ("n_neg", "n_pos", "tie_step"),
        [
            (30, 30, None),
            (50, 500, None),  # unbalanced
            (200, 200, 0.1),  # heavy ties
            (7, 311, None),  # tiny vs large
        ],
    )
    def test_matches_quadratic_reference(self, n_neg, n_pos, tie_step):
        neg, pos = binormal_scores(n_neg, n_pos, seed=42, tie_step=tie_step)
        grid = make_grid(n_neg)
        np.testing.assert_array_equal(
            empirical_roc_on_grid(neg, pos, grid), roc_on_grid_reference(neg, pos, grid)
        )

    def test_all_tied_scores_single_step(self):
        # degenerate extreme: constant scores in both classes must not fail
        neg = np.zeros(10)
        pos = np.zeros(10)
        grid = np.linspace(0, 1, 5)
        out = empirical_roc_on_grid(neg, pos, grid)
        np.testing.assert_array_equal(out, roc_on_grid_reference(neg, pos, grid))

    def test_infinite_scores_are_legal(self):
        neg = np.array([-np.inf, 0.0, 1.0])
        pos = np.array([0.5, np.inf, np.inf])
        grid = np.linspace(0, 1, 7)
        out = empirical_roc_on_grid(neg, pos, grid)
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out, roc_on_grid_reference(neg, pos, grid))

    def test_perfect_separation_hits_corner(self):
        # smallest reachable nonzero FPR with 2 negatives is 0.5; TPR is 1
        # there and beyond, 0 before (step convention, no interpolation)
        neg, pos = np.array([0.0, 0.1]), np.array([5.0, 6.0])
        grid = np.linspace(0, 1, 11)
        out = empirical_roc_on_grid(neg, pos, grid)
        assert (out[grid >= 0.5] == 1.0).all()
        assert (out[grid < 0.5] == 0.0).all()

    def test_duplicate_fpr_vertices_resolve_to_largest_tpr(self):
        # two tied negatives create duplicated FPR vertices; side="right"
        # lookup must take the largest TPR at that FPR
        neg = np.array([1.0, 1.0, 3.0])
        pos = np.array([2.0, 2.0, 4.0])
        out = empirical_roc_on_grid(neg, pos, np.array([1.0 / 3.0 + 1e-12, 1.0]))
        assert out[0] == pytest.approx(1.0 / 3.0)
        assert out[1] == 1.0

    def test_hand_worked_example_exact(self):
        # 4 negatives at 1..4, positives interleaved at 1.5..4.5: thresholds
        # (descending) 4, 3, 2, 1 give vertices (0.25, 0.25) .. (1, 1), a
        # perfect staircase. Every grid value below is derived by hand.
        neg = np.array([1.0, 2.0, 3.0, 4.0])
        pos = np.array([1.5, 2.5, 3.5, 4.5])
        grid = np.linspace(0, 1, 9)
        expected = [0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0]
        assert empirical_roc_on_grid(neg, pos, grid).tolist() == expected

    def test_rank_invariance_under_monotone_transform(self):
        # the empirical ROC only sees score order, so arctan (strictly
        # monotone) must leave every grid value bit-identical
        neg, pos = binormal_scores(60, 45, seed=9, tie_step=0.25)
        grid = make_grid(60)
        np.testing.assert_array_equal(
            empirical_roc_on_grid(np.arctan(neg), np.arctan(pos), grid),
            empirical_roc_on_grid(neg, pos, grid),
        )

    def test_vertices_are_nondecreasing(self):
        neg, pos = binormal_scores(80, 60, seed=7)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        assert (np.diff(fpr_v) >= 0).all()
        assert (np.diff(tpr_v) >= 0).all()


class TestStepLookup:
    def test_right_continuity_at_vertices(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.1, 0.7, 1.0])
        assert step_lookup(x, y, np.array([0.5]))[0] == 0.7  # jumps at x
        assert step_lookup(x, y, np.array([0.499999]))[0] == 0.1

    def test_clips_below_first_vertex(self):
        x = np.array([0.2, 1.0])
        y = np.array([0.3, 1.0])
        assert step_lookup(x, y, np.array([0.0]))[0] == 0.3


class TestKIndices:
    def test_exact_mapping_small(self):
        assert grid_k_indices(np.linspace(0, 1, 5), 4).tolist() == [0, 1, 2, 3, 4]

    def test_sentinel_only_at_one(self):
        grid = make_grid(100)
        k = grid_k_indices(grid, 100)
        assert k[-1] == 100  # -inf sentinel
        assert (k[:-1] < 100).all()

    def test_nondecreasing_and_bounded(self):
        grid = make_grid(997, grid_size=512)
        k = grid_k_indices(grid, 997)
        assert (np.diff(k.astype(np.int64)) >= 0).all()
        assert k[0] == 0
        assert int(k[-1]) == 997  # ascending, so the last entry is the max

    def test_dtype_is_uint64(self):
        assert grid_k_indices(make_grid(10), 10).dtype == np.uint64

    def test_grid_finer_than_negatives_duplicates_indices(self):
        # 11 grid points over 5 negatives: floor(t * 5) repeats every index —
        # the kernel must be exercised with duplicated thresholds, so the
        # mapping itself is pinned here by hand
        k = grid_k_indices(np.linspace(0, 1, 11), n_neg=5)
        assert k.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
