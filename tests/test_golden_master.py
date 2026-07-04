"""Golden-master equivalence — the highest-value suite.

Given identical ``(boot_tpr_matrix, fpr_grid, y_true, y_score, alpha)``
fixtures, the rocci assembly must reproduce the recorded band outputs
within ``atol=1e-6`` (float32 fixtures widened to float64).

Precedence rule: if this test disagrees with the implementation, the
committed golden-master fixture wins — never regenerate fixtures to match
new code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rocci.band.envelope import assemble_envelope_band, studentized_envelope
from rocci.band.floors import rectangle_floor
from rocci.band.grids import empirical_roc_on_grid

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
FIXTURES = sorted(GOLDEN_DIR.glob("*.npz"))
ATOL = 1e-6


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def split_classes(fx):
    y_true = fx["y_true"].astype(np.int64)
    y_score = fx["y_score"].astype(np.float64)
    return y_score[y_true == 0], y_score[y_true == 1]


@pytest.mark.skipif(not FIXTURES, reason="golden-master fixtures not recorded yet")
class TestGoldenMasters:
    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
    def test_full_band_matches_paper_implementation(self, path):
        fx = load(path)
        neg, pos = split_classes(fx)
        band = assemble_envelope_band(
            fx["boot_tpr"].astype(np.float64),
            fx["fpr_grid"].astype(np.float64),
            neg,
            pos,
            float(fx["alpha"]),
        )
        # Documented statistical delta from the recorded implementation
        # (spec §5.6 / appendix A9): rocci additionally pins lower[-1] = 1.0
        # because the true ROC at FPR = 1 is identically 1, so the pin
        # tightens the band at zero coverage cost. The fixtures are NOT
        # regenerated; every other grid point must still match the validated
        # implementation exactly.
        np.testing.assert_allclose(
            band.lower[:-1],
            fx["lower"].astype(np.float64)[:-1],
            atol=ATOL,
            err_msg="lower band diverged from the validated implementation",
        )
        assert band.lower[-1] == 1.0, "final lower endpoint must be pinned to 1"
        np.testing.assert_allclose(
            band.upper,
            fx["upper"].astype(np.float64),
            atol=ATOL,
            err_msg="upper band diverged from the validated implementation",
        )

    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
    def test_pre_floor_arm_matches(self, path):
        # isolates studentization + retention + envelope before floor application
        fx = load(path)
        if "pre_floor_lower" not in fx:
            pytest.skip("fixture lacks pre-floor arm")
        neg, pos = split_classes(fx)
        grid = fx["fpr_grid"].astype(np.float64)
        boot = fx["boot_tpr"].astype(np.float64)
        tpr_hat = empirical_roc_on_grid(neg, pos, grid)
        lo, hi, _, _ = studentized_envelope(
            boot, tpr_hat, float(fx["alpha"]), len(neg), len(pos)
        )
        lo, hi = lo.copy(), hi.copy()
        lo[0] = 0.0  # endpoints are pinned on every variant
        hi[-1] = 1.0
        np.testing.assert_allclose(lo, fx["pre_floor_lower"], atol=ATOL)
        np.testing.assert_allclose(hi, fx["pre_floor_upper"], atol=ATOL)

    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
    def test_rectangle_floored_arm_matches(self, path):
        # isolates envelope with variance-ratio gated Wilson rectangle floor
        fx = load(path)
        if "no_beta_lower" not in fx:
            pytest.skip("fixture lacks no-beta-floor arm")
        neg, pos = split_classes(fx)
        grid = fx["fpr_grid"].astype(np.float64)
        boot = fx["boot_tpr"].astype(np.float64)
        alpha = float(fx["alpha"])
        tpr_hat = empirical_roc_on_grid(neg, pos, grid)
        lo_env, hi_env, var_raw, wilson_var = studentized_envelope(
            boot, tpr_hat, alpha, len(neg), len(pos)
        )
        lo, hi = rectangle_floor(
            lo_env,
            hi_env,
            var_raw=var_raw,
            wilson_var=wilson_var,
            neg=neg,
            pos=pos,
            grid=grid,
            alpha=alpha,
        )
        lo, hi = lo.copy(), hi.copy()
        lo[0] = 0.0
        hi[-1] = 1.0
        np.testing.assert_allclose(lo, fx["no_beta_lower"], atol=ATOL)
        np.testing.assert_allclose(hi, fx["no_beta_upper"], atol=ATOL)
