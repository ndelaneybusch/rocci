"""Studentized envelope, assembly + attribution, and AUC calculation.

Risks mitigated: retention off-by-one or tie mishandling; the collapse
guard silently dividing by ~0; assembly reordering; attribution codes not
reflecting which mechanism actually set the lower band.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocci.backend._fallback import bootstrap_tpr_matrix_numpy
from rocci.band.envelope import (
    ATTR_BETA_FLOOR,
    ATTR_BOOTSTRAP,
    ATTR_PINNED,
    ATTR_WILSON_FLOOR,
    assemble_envelope_band,
    bootstrap_auc_ci,
    kernel_grid_auc,
    mann_whitney_auc,
    studentized_envelope,
)
from rocci.band.floors import beta_floor_vacuous_below
from rocci.band.grids import grid_k_indices, make_grid
from tests.conftest import binormal_scores


def build_band(n_neg=60, n_pos=60, auc=0.8, seed=0, n_boot=600, alpha=0.05, **kw):
    neg, pos = binormal_scores(n_neg, n_pos, auc=auc, seed=seed, **kw)
    neg, pos = np.sort(neg), np.sort(pos)
    grid = make_grid(n_neg)
    boot = bootstrap_tpr_matrix_numpy(
        neg, pos, grid_k_indices(grid, n_neg), n_boot=n_boot, seed=seed + 1
    )
    return assemble_envelope_band(boot, grid, neg, pos, alpha), neg, pos, grid


class TestStudentizedEnvelope:
    def test_retention_keeps_most_typical_curves(self):
        # 10 constant curves at increasing distance from a flat empirical
        # ROC: with alpha=0.3, ceil(0.7*10)=7 closest survive; the envelope
        # is exactly their min/max
        tpr_hat = np.full(4, 0.5)
        offsets = np.array(
            [0.00, 0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.5, 0.4]
        )
        boot = 0.5 + np.tile(offsets[:, None], (1, 4))
        lo, hi, _, _ = studentized_envelope(boot, tpr_hat, 0.3, 50, 50)
        assert lo[0] == pytest.approx(0.5 - 0.06)
        assert hi[0] == pytest.approx(0.5 + 0.05)

    def test_ties_at_threshold_keep_extra_curves(self):
        tpr_hat = np.full(3, 0.5)
        # curves at |offset|: 0, .01, .02, .03(x3 tied), .5 — ceil(.7*7)=5
        # ranked stats: the 5th is one of the tied .03s, so all three stay
        offsets = np.array([0.00, 0.01, 0.02, 0.03, 0.03, -0.03, 0.5])
        boot = 0.5 + np.tile(offsets[:, None], (1, 3))
        lo, hi, _, _ = studentized_envelope(boot, tpr_hat, 0.3, 50, 50)
        assert lo[0] == pytest.approx(0.5 - 0.03)
        assert hi[0] == pytest.approx(0.5 + 0.03)

    def test_collapse_guard_scores_noise_as_zero(self):
        # n_pos=0 zeroes the Wilson floor; constant columns give sd=0 < eps,
        # so tiny deviations score 0 and the curve is retained
        tpr_hat = np.array([0.5, 0.5])
        boot = np.full((20, 2), 0.5)
        boot[0, 0] += 1e-9  # numerical noise, below eps
        lo, hi, _var_raw, wilson_var = studentized_envelope(boot, tpr_hat, 0.05, 50, 0)
        assert (wilson_var == 0).all()
        assert lo[0] <= 0.5 <= hi[0]

    def test_collapse_guard_scores_real_deviation_against_eps(self):
        tpr_hat = np.array([0.5, 0.5])
        boot = np.full((100, 2), 0.5)
        # eps = 1/(50+0) = 0.02; a 0.03 deviation in a 100-row column keeps
        # the column sd (~0.003) under eps, forcing the eps-branch scoring
        boot[0, 0] = 0.53
        _lo, hi, _, _ = studentized_envelope(boot, tpr_hat, 0.05, 50, 0)
        # that replicate scores 0.03/eps = 1.5 > 0 and is trimmed
        assert hi[0] == pytest.approx(0.5)

    def test_envelope_clipped_to_unit_interval(self):
        tpr_hat = np.full(3, 0.5)
        rng = np.random.default_rng(0)
        boot = np.clip(0.5 + rng.normal(0, 0.4, (300, 3)), -0.2, 1.2)
        lo, hi, _, _ = studentized_envelope(boot, tpr_hat, 0.05, 20, 20)
        assert (lo >= 0).all()
        assert (hi <= 1).all()


class TestAssembly:
    def test_final_band_ordered_and_monotone(self):
        band, *_ = build_band()
        assert (band.lower <= band.upper + 1e-12).all()
        assert (np.diff(band.lower) >= -1e-12).all()
        assert (np.diff(band.upper) >= -1e-12).all()

    def test_endpoints_pinned(self):
        band, *_ = build_band(seed=1)
        assert band.lower[0] == 0.0
        assert band.upper[-1] == 1.0
        assert band.attribution[0] == ATTR_PINNED

    def test_attribution_codes_reflect_mechanisms(self):
        band, *_ = build_band(n_neg=100, n_pos=100, auc=0.95, seed=2)
        attr = band.attribution
        assert set(np.unique(attr)) <= {
            ATTR_BOOTSTRAP,
            ATTR_BETA_FLOOR,
            ATTR_WILSON_FLOOR,
            ATTR_PINNED,
        }
        tol = 1e-12
        beta_pts = attr == ATTR_BETA_FLOOR
        assert (band.lower[beta_pts] < band.lower_rect[beta_pts] - tol).all()
        wilson_pts = attr == ATTR_WILSON_FLOOR
        assert (band.lower_rect[wilson_pts] < band.lower_env[wilson_pts] - tol).all()
        boot_pts = attr == ATTR_BOOTSTRAP
        np.testing.assert_allclose(band.lower[boot_pts], band.lower_env[boot_pts])

    def test_high_auc_engages_beta_floor(self):
        # at AUC 0.95 the bootstrap collapses near FPR=0 and the exact Beta
        # floor must take over part of the low-FPR lower band
        band, *_ = build_band(n_neg=200, n_pos=200, auc=0.95, seed=3, n_boot=800)
        assert (band.attribution == ATTR_BETA_FLOOR).any()

    def test_vacuous_region_lower_band_is_zero(self):
        band, _, _, grid = build_band(seed=4)
        q1 = beta_floor_vacuous_below(60, 0.05)
        assert band.vacuous_below == pytest.approx(q1)
        in_vac = (grid > 0) & (grid < q1)
        assert (band.lower[in_vac] == 0.0).all()

    def test_floors_never_narrow_the_envelope_upper(self):
        band, *_ = build_band(seed=5)
        assert (band.upper >= band.upper_env - 1e-12).all()

    def test_band_result_is_frozen(self):
        band, *_ = build_band(seed=6)
        with pytest.raises(AttributeError):
            band.alpha = 0.5

    def test_heavy_ties_still_assemble(self):
        band, *_ = build_band(seed=7, tie_step=0.25)
        assert (band.lower <= band.upper + 1e-12).all()

    def test_constant_scores_degenerate_to_floors(self):
        neg = np.zeros(30)
        pos = np.zeros(30)
        grid = make_grid(30)
        boot = bootstrap_tpr_matrix_numpy(
            neg, pos, grid_k_indices(grid, 30), n_boot=200, seed=0
        )
        band = assemble_envelope_band(boot, grid, neg, pos, 0.05)
        assert (band.lower <= band.upper + 1e-12).all()
        assert band.upper[-1] == 1.0


class TestAuc:
    @staticmethod
    def brute_force_mw(neg, pos):
        """Literal pairwise Mann-Whitney oracle, ties weighted 1/2."""
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        return (gt + 0.5 * eq) / (len(neg) * len(pos))

    @pytest.mark.parametrize(
        "tie_step", [None, 0.5, 100.0], ids=["continuous", "heavy_ties", "all_tied"]
    )
    def test_auc_matches_bruteforce_mann_whitney(self, tie_step):
        neg, pos = binormal_scores(150, 130, seed=8, tie_step=tie_step)
        assert mann_whitney_auc(neg, pos) == pytest.approx(
            self.brute_force_mw(neg, pos), abs=1e-12
        )

    def test_auc_handles_hand_checked_ties(self):
        neg = np.array([0.0, 1.0, 1.0])
        pos = np.array([1.0, 2.0])
        # pairs: (1,0)> (1,1)= (1,1)= (2,0)> (2,1)> (2,1)>  -> (4 + 0.5*2)/6
        assert mann_whitney_auc(neg, pos) == pytest.approx(5.0 / 6.0)

    def test_bootstrap_ci_brackets_point_estimate(self):
        neg, pos = binormal_scores(100, 100, seed=9)
        neg, pos = np.sort(neg), np.sort(pos)
        grid = make_grid(100)
        boot = bootstrap_tpr_matrix_numpy(
            neg, pos, grid_k_indices(grid, 100), n_boot=500, seed=10
        )
        lo, hi = bootstrap_auc_ci(boot, grid, neg, pos, alpha=0.05)
        auc = mann_whitney_auc(neg, pos)
        assert lo < auc < hi
        assert 0.0 <= lo <= hi <= 1.0

    @pytest.mark.parametrize("tie_step", [1.0, 100.0], ids=["integer", "binary_like"])
    def test_ci_brackets_point_estimate_under_heavy_ties(self, tie_step):
        # the raw percentile CI of strictly-greater grid AUCs sits below the
        # Mann-Whitney point estimate by ~half the tie mass; the recentering
        # must repair exactly this failure (appendix A10 delta note)
        neg, pos = binormal_scores(150, 150, seed=11, tie_step=tie_step)
        neg, pos = np.sort(neg), np.sort(pos)
        grid = make_grid(150)
        boot = bootstrap_tpr_matrix_numpy(
            neg, pos, grid_k_indices(grid, 150), n_boot=500, seed=12
        )
        lo, hi = bootstrap_auc_ci(boot, grid, neg, pos, alpha=0.05)
        assert lo <= mann_whitney_auc(neg, pos) <= hi

    def test_ci_percentiles_use_linear_method(self):
        grid = np.linspace(0, 1, 3)
        boot = np.array([[0, a, 1] for a in (0.1, 0.2, 0.3, 0.4)], dtype=float)
        neg = np.array([0.0, 1.0])
        pos = np.array([2.0, 3.0])  # separable: MW = kernel plug-in = 1
        assert mann_whitney_auc(neg, pos) == kernel_grid_auc(neg, pos, grid) == 1.0
        aucs = 0.25 + np.array([0.1, 0.2, 0.3, 0.4]) / 2  # trapezoid by hand
        lo, hi = bootstrap_auc_ci(boot, grid, neg, pos, alpha=0.5)
        assert lo == pytest.approx(np.percentile(aucs, 25))  # zero recentering
        assert hi == pytest.approx(np.percentile(aucs, 75))
