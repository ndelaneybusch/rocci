"""Wilson confidence bounds, rectangle band, variance gate, and Beta floor.

The floors are the safety net that keeps the band honest where the bootstrap
under-reports variance (near FPR = 0 at high AUC, and wherever a resampled TPR
column collapses). This suite pins each floor to its closed form by independent
derivation — not a scipy round-trip — and locks down the two invariants that make
a floor safe: it may only *widen* the band, and it may only act inside its own
jurisdiction.

Guaranteed. The Wilson endpoints are checked against the score-test inversion
they solve, ``(p_hat - p)^2 = z^2 p(1-p)/n``, so a drift in the closed form
breaks the defining identity (and the one-sided lower bound uses the one-sided
z). The variance-gated rectangle floor leaves healthy-variance points untouched,
only ever widens deficient ones, keeps the output monotone, uses the Sidak
correction only when the effective count ``k_eff > 1`` (and the uncorrected alpha
otherwise), and matches a hand-transcribed Sidak oracle under full collapse. The
Beta order-statistic floor is provably vacuous below ``q_1`` (with
``q_1 = 1 - (alpha/(2 j_max))^(1/n_neg)`` confirmed in closed form and via the
Beta/Binomial survival identity that catches swapped parameters), never raises
the band, is a no-op outside its ``(0, q_max]`` zone and on empty classes, and
its floored value equals the one-sided Wilson lower bound of a hand-counted TPR.

Limitations. These are exact primitive-level checks with synthetic envelope
inputs; how the floors compose into the final band (assembly order, attribution)
is ``test_envelope.py``, and their statistical effect on coverage is the
statistics/calibration suites.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import binom, norm

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

    @pytest.mark.parametrize("p_hat", [0.02, 0.3, 0.5, 0.77, 0.98])
    @pytest.mark.parametrize(("n", "z"), [(5, 1.0), (40, 1.96), (1000, 3.0)])
    def test_bounds_satisfy_defining_equation(self, p_hat, n, z):
        # first principles: the Wilson endpoints are exactly the two roots p of
        # (p_hat - p)^2 = z^2 p(1-p)/n — the score-test inversion that defines
        # the interval. Any drift in the closed form breaks this identity.
        lo, hi = wilson_bounds(np.array([p_hat]), n, z)
        for p in (lo[0], hi[0]):
            assert (p_hat - p) ** 2 == pytest.approx(z * z * p * (1 - p) / n, abs=1e-13)

    def test_one_sided_lower_satisfies_defining_equation(self):
        p_hat, n, alpha = 0.8, 50, 0.05
        z1 = float(norm.ppf(1 - alpha))
        p = wilson_lower_one_sided(np.array([p_hat]), n, alpha)[0]
        assert (p_hat - p) ** 2 == pytest.approx(z1 * z1 * p * (1 - p) / n, abs=1e-13)

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

    def test_full_collapse_matches_sidak_oracle(self):
        # var_raw = 0 everywhere gives deficiency 1 at every grid point, so
        # k_eff = K and alpha_w = 1 - (1-alpha)^(1/K) exactly; the floored
        # band must equal min/max against the rectangle band at alpha_w
        # followed by the monotonicity pass — transcribed here independently
        # of the implementation
        neg, pos, grid, tpr, wilson_var = self._setup(seed=13)
        lo_env = np.clip(tpr - 0.02, 0, 1)
        hi_env = np.clip(tpr + 0.02, 0, 1)
        lo, hi = rectangle_floor(
            lo_env,
            hi_env,
            var_raw=np.zeros_like(grid),
            wilson_var=wilson_var,
            neg=neg,
            pos=pos,
            grid=grid,
            alpha=0.05,
        )
        alpha_w = 1.0 - 0.95 ** (1.0 / len(grid))
        rect_lo, rect_hi = wilson_rectangle_band(neg, pos, grid, alpha_w)
        exp_lo = np.minimum(lo_env, rect_lo)
        exp_hi = np.maximum(hi_env, rect_hi)
        exp_hi = np.maximum.accumulate(exp_hi)
        exp_lo = np.minimum.accumulate(exp_lo[::-1])[::-1]
        np.testing.assert_array_equal(lo, exp_lo)
        np.testing.assert_array_equal(hi, exp_hi)

    def test_k_eff_at_most_one_uses_uncorrected_alpha(self):
        # a single point with deficiency 0.4 gives k_eff = 0.4 <= 1, so the
        # rectangle floor must be evaluated at the *uncorrected* alpha (the
        # Sidak branch fires only for k_eff > 1)
        neg, pos, grid, tpr, wilson_var = self._setup(seed=14)
        lo_env = np.clip(tpr - 0.02, 0, 1)
        hi_env = np.clip(tpr + 0.02, 0, 1)
        var_raw = wilson_var * 2.0
        j = 25
        var_raw[j] = 0.6 * wilson_var[j]  # deficiency exactly 0.4
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
        rect_lo, rect_hi = wilson_rectangle_band(neg, pos, grid, alpha=0.05)
        exp_lo, exp_hi = lo_env.copy(), hi_env.copy()
        exp_lo[j] = min(exp_lo[j], rect_lo[j])
        exp_hi[j] = max(exp_hi[j], rect_hi[j])
        exp_hi = np.maximum.accumulate(exp_hi)
        exp_lo = np.minimum.accumulate(exp_lo[::-1])[::-1]
        assert lo[j] < lo_env[j], "floor must engage at the deficient point"
        np.testing.assert_array_equal(lo, exp_lo)
        np.testing.assert_array_equal(hi, exp_hi)


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
        # split stays Bonferroni over 2*j_max
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

    @pytest.mark.parametrize(("n_neg", "alpha"), [(2, 0.05), (80, 0.05), (500, 0.01)])
    def test_vacuous_below_closed_form(self, n_neg, alpha):
        # Beta(1, n) has CDF 1 - (1-x)^n, so q_1 = 1 - a_e^(1/n) in closed
        # form — an independent derivation, not a scipy round-trip
        a_e = alpha / (2 * J_MAX)
        expected = 1.0 - a_e ** (1.0 / n_neg)
        assert beta_floor_vacuous_below(n_neg, alpha) == pytest.approx(
            expected, rel=1e-12
        )

    def test_jurisdiction_edges_satisfy_binomial_identity(self):
        # first principles: the FPR exceedance at the j-th largest of n0
        # negatives is Beta(j, n0+1-j), whose survival function satisfies
        # P(Beta > x) = P(Bin(n0, x) <= j-1). The jurisdiction edges q_j are
        # defined by P(Beta > q_j) = a_e, so binom.cdf(j-1, n0, q_j) must
        # equal a_e — this catches swapped Beta parameters, which the ppf
        # round-trip cannot.
        n0, alpha = 40, 0.05
        a_e = alpha / (2 * J_MAX)
        js = np.arange(1, J_MAX + 1)
        q = beta_dist.ppf(1 - a_e, js, n0 + 1 - js)
        np.testing.assert_allclose(binom.cdf(js - 1, n0, q), a_e, rtol=1e-9)
        assert (np.diff(q) > 0).all(), "jurisdiction edges must be increasing"

    def test_floor_value_is_wilson_bound_of_hand_counted_tpr(self):
        # hand-checkable construction: 30 negatives 0..29; positives place
        # 2 of 4 strictly above the largest negative (tpr_1 = 1/2) and all 4
        # above the 2nd largest (tpr_2 = 1). Inside (q_1, q_2) the floor must
        # be the one-sided Wilson lower bound of 1/2; at t = q_2 and beyond
        # (side='right' lookup) the bound of 1.
        neg = np.arange(30, dtype=float)
        pos = np.array([28.5, 28.5, 29.5, 29.5])
        alpha = 0.05
        a_e = alpha / (2 * J_MAX)
        js = np.arange(1, J_MAX + 1)
        q = beta_dist.ppf(1 - a_e, js, 30 + 1 - js)
        grid = np.array([q[0] / 2, 0.5 * (q[0] + q[1]), q[1], 0.5 * (q[1] + q[2])])
        floored = beta_orderstat_floor(grid, np.ones_like(grid), neg, pos, alpha)
        wl_half = wilson_lower_one_sided(np.array([0.5]), 4, a_e)[0]
        wl_one = wilson_lower_one_sided(np.array([1.0]), 4, a_e)[0]
        assert floored[0] == 0.0  # below q_1: provably vacuous
        assert floored[1] == wl_half
        assert floored[2] == wl_one  # boundary belongs to the larger j
        assert floored[3] == wl_one

    def test_grid_zero_is_outside_the_zone(self):
        # the floor's zone is (0, q_max]: t = 0 must be left untouched even
        # though it is below q_1 (the pinned endpoint owns it)
        neg, pos = binormal_scores(50, 50, seed=15)
        grid = np.array([0.0, 0.001])
        base = np.array([0.9, 0.9])
        floored = beta_orderstat_floor(grid, base, neg, pos, 0.05)
        assert floored[0] == 0.9
        assert floored[1] == 0.0
