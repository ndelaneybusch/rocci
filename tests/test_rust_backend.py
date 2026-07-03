"""Rust backend contract: determinism and cross-backend agreement.

Risks mitigated: RNG streams accidentally depending on thread scheduling
(silently breaking reproducibility), and the two kernels drifting apart
statistically while both looking individually plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

import rocci.backend as backend_mod
from rocci.backend import BACKEND
from rocci.backend._fallback import bootstrap_tpr_matrix_numpy
from rocci.band.grids import grid_k_indices, make_grid
from tests.conftest import binormal_scores

pytestmark = pytest.mark.skipif(
    BACKEND != "rust", reason="Rust core not available in this environment"
)


def _data(n_neg=200, n_pos=200, seed=0, grid_size=None):
    neg, pos = binormal_scores(n_neg, n_pos, seed=seed)
    neg, pos = np.sort(neg), np.sort(pos)
    grid = make_grid(n_neg, grid_size=grid_size)
    return neg, pos, grid_k_indices(grid, n_neg), grid


class TestDeterminism:
    @pytest.mark.parametrize("n_threads", [1, 4, None])
    def test_bit_identical_across_thread_counts(self, n_threads):
        neg, pos, k, _ = _data()
        reference = backend_mod.bootstrap_tpr_matrix(neg, pos, k, 500, 7, n_threads=1)
        out = backend_mod.bootstrap_tpr_matrix(neg, pos, k, 500, 7, n_threads=n_threads)
        np.testing.assert_array_equal(out, reference)

    def test_bit_identical_across_runs(self):
        neg, pos, k, _ = _data(seed=1)
        a = backend_mod.bootstrap_tpr_matrix(neg, pos, k, 300, 11)
        b = backend_mod.bootstrap_tpr_matrix(neg, pos, k, 300, 11)
        np.testing.assert_array_equal(a, b)

    def test_sentinel_column_is_one(self):
        neg, pos, k, _ = _data(n_neg=50, n_pos=50)
        out = backend_mod.bootstrap_tpr_matrix(neg, pos, k, 100, 3)
        assert (out[:, -1] == 1.0).all()  # k = n_neg sentinel at t = 1

    def test_values_are_multiples_of_inv_n_pos(self):
        neg, pos, k, _ = _data(n_neg=40, n_pos=37)
        out = backend_mod.bootstrap_tpr_matrix(neg, pos, k, 100, 5)
        np.testing.assert_allclose(out * 37, np.round(out * 37), atol=1e-9)


@pytest.mark.slow
class TestCrossBackendAgreement:
    """Verify distributional agreement between backends at B=8000:
    max pointwise mean-difference z < 6 and interior std ratios within [0.9, 1.1]."""

    def test_distributional_agreement(self):
        n = 200
        b = 8000
        neg, pos, k, grid = _data(n_neg=n, n_pos=n, seed=2)
        rust = backend_mod.bootstrap_tpr_matrix(neg, pos, k, b, 123)
        numpy_ = bootstrap_tpr_matrix_numpy(neg, pos, k, b, 456)

        mean_r, mean_n = rust.mean(axis=0), numpy_.mean(axis=0)
        var_r, var_n = rust.var(axis=0, ddof=1), numpy_.var(axis=0, ddof=1)
        se = np.sqrt(var_r / b + var_n / b)
        z = np.abs(mean_r - mean_n) / np.where(se > 0, se, np.inf)
        assert z.max() < 6.0, f"max mean-difference z = {z.max():.2f}"

        interior = (grid >= 0.1) & (grid <= 0.9)
        ratio = np.sqrt(var_r[interior]) / np.sqrt(var_n[interior])
        assert ratio.min() > 0.9, f"std ratio min {ratio.min():.3f}"
        assert ratio.max() < 1.1, f"std ratio max {ratio.max():.3f}"
