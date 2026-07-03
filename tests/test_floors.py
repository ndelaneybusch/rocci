"""Wilson machinery (A4), rectangle band (A5), gate (A7), Beta floor (A8).

Risks mitigated: floor formulas drifting from the closed forms; the
rectangle floor narrowing instead of widening; the Beta floor raising the
band or claiming jurisdiction it does not have.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from rocci.band.floors import (
    J_MAX,
    beta_floor_vacuous_below,
    beta_orderstat_floor,
    rectangle_floor,
    wilson_bounds,
    wilson_halfwidth_sq,
    wilson_lower_one_sided,
    wilson_rectangle_band,
)
from rocci.band.grids import empirical_roc_on_grid, make_grid
from tests.conftest import binormal_scores


class TestWilsonClosedForms:
    def test_matches_scalar_formula(self):
        p, n, z = 0.3, 40, 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        lo, hi = wilson_bounds(np.array([p]), n, z)
        assert lo[0] == pytest.approx(center - half, abs=1e-15)
        assert hi[0] == pytest.approx(center + half, abs=1e-15)

    def test_nonzero_width_at_boundaries(self):
        lo, hi = wilson_bounds(np.array([0.0, 1.0]), 100, 1.96)
        assert hi[0] > 0.0  # p=0 still has upper mass
        assert lo[1] < 1.0  # p=1 still has lower mass

    def test_degenerate_n_returns_vacuous_interval(self):
        lo, hi = wilson_bounds(np.array([0.5]), 0, 1.96)
        assert lo[0] == 0.0
        assert hi[0] == 1.0

    def test_halfwidth_sq_positive_even_at_boundaries(self):
        v = wilson_halfwidth_sq(np.array([0.0, 0.5, 1.0]), 25, 2.24)
        assert (v > 0).all()

    def test_halfwidth_sq_is_squared_halfwidth(self):
        p, n, z = np.array([0.42]), 60, 1.96
        lo, hi = wilson_bounds(p, n, z)
        # away from the clip region, (hi - lo)/2 squared == halfwidth_sq
        assert wilson_halfwidth_sq(p, n, z)[0] == pytest.approx(
            ((hi[0] - lo[0]) / 2) ** 2, rel=1e-12
        )

    def test_halfwidth_sq_degenerate_n_is_zero(self):
        assert (wilson_halfwidth_sq(np.array([0.5]), 0, 1.96) == 0).all()

    def test_one_sided_lower_uses_one_sided_z(self):
        p, n, alpha = np.array([0.8]), 50, 0.05
        z1 = norm.ppf(1 - alpha)
        denom = 1 + z1 * z1 / n
        center = (p[0] + z1 * z1 / (2 * n)) / denom
        half = (z1 / denom) * np.sqrt(p[0] * (1 - p[0]) / n + z1 * z1 / (4 * n * n))
        assert wilson_lower_one_sided(p, n, alpha)[0] == pytest.approx(
            center - half, abs=1e-15
        )


class TestWilsonRectangleBand:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_band_contains_empirical_curve_interior(self, seed):
        neg, pos = binormal_scores(80, 80, seed=seed)
        grid = make_grid(80)
        lo, hi = wilson_rectangle_band(neg, pos, grid, alpha=0.05)
        tpr = empirical_roc_on_grid(neg, pos, grid)
        interior = (grid > 0.05) & (grid < 0.95)
        assert (lo[interior] <= tpr[interior] + 1e-12).all()
        assert (hi[interior] >= tpr[interior] - 1e-12).all()

    def test_endpoints_pinned(self):
        neg, pos = binormal_scores(30, 30, seed=3)
        lo, hi = wilson_rectangle_band(neg, pos, make_grid(30), alpha=0.05)
        assert lo[0] == 0.0
        assert hi[-1] == 1.0

    def test_bounds_ordered_and_in_unit_interval(self):
        neg, pos = binormal_scores(50, 500, seed=4)
        lo, hi = wilson_rectangle_band(neg, pos, make_grid(50), alpha=0.05)
        assert (lo >= 0).all()
        assert (hi <= 1).all()
        assert (lo <= hi + 1e-12).all()


class TestRectangleFloorGate:
    def _setup(self, seed=5, n=60):
        neg, pos = binormal_scores(n, n, seed=seed)
        grid = make_grid(n)
        tpr = empirical_roc_on_grid(neg, pos, grid)
        z = float(norm.ppf(0.975))
        wilson_var = wilson_halfwidth_sq(tpr, n, z) / z**2
        return neg, pos, grid, tpr, wilson_var

    def test_healthy_variance_is_untouched(self):
        neg, pos, grid, tpr, wilson_var = self._setup()
        lo_env, hi_env = tpr - 0.1, tpr + 0.1
        lo, hi = rectangle_floor(
            lo_env,
            hi_env,
            var_raw=wilson_var * 2.0,  # everywhere above the Wilson floor
            wilson_var=wilson_var,
            neg=neg,
            pos=pos,
            grid=grid,
            alpha=0.05,
        )
        np.testing.assert_array_equal(lo, lo_env)
        np.testing.assert_array_equal(hi, hi_env)

    def test_floor_only_widens(self):
        neg, pos, grid, tpr, wilson_var = self._setup()
        lo_env = np.clip(tpr - 0.02, 0, 1)
        hi_env = np.clip(tpr + 0.02, 0, 1)
        lo, hi = rectangle_floor(
            lo_env,
            hi_env,
            var_raw=np.zeros_like(grid),  # fully collapsed everywhere
            wilson_var=wilson_var,
            neg=neg,
            pos=pos,
            grid=grid,
            alpha=0.05,
        )
        assert (lo <= lo_env + 1e-12).all()
        assert (hi >= hi_env - 1e-12).all()

    def test_output_bands_are_monotone(self):
        neg, pos, grid, tpr, wilson_var = self._setup(seed=6)
        rng = np.random.default_rng(0)
        lo_env = np.clip(tpr - rng.uniform(0, 0.05, len(grid)), 0, 1)
        hi_env = np.clip(tpr + rng.uniform(0, 0.05, len(grid)), 0, 1)
        var_raw = wilson_var.copy()
        var_raw[::3] = 0.0  # collapse a scattered subset of points
        lo, hi = rectangle_floor(
            lo_env,
            hi_env,
            var_raw=var_raw,
            wilson_var=wilson_var,
            neg=neg,
            pos=pos,
            grid=grid,
            alpha=0.05,
        )
        assert (np.diff(lo) >= -1e-12).all()
        assert (np.diff(hi) >= -1e-12).all()

    def test_partial_deficiency_floors_only_gated_points(self):
        neg, pos, grid, tpr, wilson_var = self._setup(seed=7)
        lo_env = np.clip(tpr - 0.02, 0, 1)  # narrow: floor engages at the gap
        hi_env = np.clip(tpr + 0.02, 0, 1)
        var_raw = wilson_var * 2.0
        j = 30  # single collapsed interior point
        var_raw[j] = 0.0
        lo, hi = rectangle_floor(
            lo_env,
            hi_env,
            var_raw=var_raw,
            wilson_var=wilson_var,
            neg=neg,
            pos=pos,
            grid=grid,
            alpha=0.05,
        )
        # floor widens at j; the monotonicity pass may propagate the lower
        # bound leftward and the upper bound rightward, but never the reverse
        assert lo[j] < lo_env[j]
        assert hi[j] > hi_env[j]
        np.testing.assert_array_equal(lo[j + 1 :], lo_env[j + 1 :])
        np.testing.assert_array_equal(hi[:j], hi_env[:j])


class TestBetaOrderstatFloor:
    def test_vacuous_below_q1(self):
        neg, pos = binormal_scores(100, 100, seed=8)
        grid = make_grid(100)
        q1 = beta_floor_vacuous_below(100, alpha=0.05)
        lower = beta_orderstat_floor(grid, np.ones_like(grid), neg, pos, 0.05)
        in_vacuous = (grid > 0) & (grid < q1)
        assert in_vacuous.any()
        assert (lower[in_vacuous] == 0.0).all()

    def test_never_raises_the_band(self):
        neg, pos = binormal_scores(100, 100, seed=9)
        grid = make_grid(100)
        base = np.full_like(grid, 0.4)
        floored = beta_orderstat_floor(grid, base, neg, pos, 0.05)
        assert (floored <= base + 1e-15).all()

    def test_no_op_outside_jurisdiction(self):
        neg, pos = binormal_scores(200, 200, seed=10)
        grid = make_grid(200)
        a_e = 0.05 / (2 * J_MAX)
        q_max = beta_dist.ppf(1 - a_e, J_MAX, 200 + 1 - J_MAX)
        base = np.full_like(grid, 0.4)
        floored = beta_orderstat_floor(grid, base, neg, pos, 0.05)
        outside = grid > q_max
        np.testing.assert_array_equal(floored[outside], base[outside])

    def test_jurisdiction_capped_by_n_neg(self):
        # n_neg < j_max: only n_neg order statistics exist, but the alpha
        # split stays Bonferroni over 2*j_max (per A8)
        neg, pos = binormal_scores(10, 50, seed=11)
        grid = make_grid(10)
        floored = beta_orderstat_floor(grid, np.ones_like(grid), neg, pos, 0.05)
        assert (floored <= 1.0).all()
        assert floored[1] < 1.0  # floor engaged despite tiny n_neg

    def test_empty_classes_are_a_no_op(self):
        grid = make_grid(10)
        base = np.full_like(grid, 0.5)
        out = beta_orderstat_floor(grid, base, np.array([]), np.array([]), 0.05)
        np.testing.assert_array_equal(out, base)

    def test_heavy_ties_floor_still_valid_shape(self):
        neg, pos = binormal_scores(150, 150, seed=12, tie_step=0.5)
        grid = make_grid(150)
        floored = beta_orderstat_floor(grid, np.ones_like(grid), neg, pos, 0.05)
        assert (floored >= 0).all()
        assert (floored <= 1).all()

    def test_vacuous_below_matches_beta_quantile(self):
        expected = beta_dist.ppf(1 - 0.05 / (2 * J_MAX), 1, 80)
        assert beta_floor_vacuous_below(80, 0.05) == pytest.approx(expected)
