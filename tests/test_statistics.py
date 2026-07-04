"""Statistical gold standards for the assembled envelope band.

A band can be built from primitives that each match their formula and still fail
as a whole — miss the true ROC, be wider than the trivial construction it exists
to beat, or stop responding to its knobs. This suite checks those whole-band
properties against ground truth (DGPs with closed-form population ROCs, appendix
A15) on single fast draws, as the quick companion to the full multi-DGP coverage
gate in ``test_calibration.py``.

Guaranteed. On the recorded draws the band contains the true binormal ROC (the
reason it exists), and it is strictly narrower than the KS/DKW reference band
(appendix A16) on both binormal and heavy-tailed data — the paper's headline
"tighter than the trivial valid band" claim, and the distribution-free part of it.
The band responds correctly to its two knobs: raising confidence nests the bands
pointwise (more confidence never buys a narrower bound) and grows the vacuous
region as theory requires, while quadrupling n shrinks the mean width at roughly
the ``sqrt(n)`` rate. And the honest cost of validity is bounded: on
correctly-specified binormal data the envelope is wider than Working-Hotelling
but by less than 2.5x — if it ever undercut the correct parametric band, it would
be over-tight and coverage would be at risk.

Limitations. All seeds are fixed, so these are deterministic regression checks on
specific draws (recorded as covered), not a stochastic coverage estimate — that
is ``test_calibration.py``. Direction and rate are asserted within tolerance
windows, not as exact values, and containment is claimed only for these draws.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from rocci import roc_band
from rocci._warnings import RocciWarning
from rocci.band.grids import empirical_roc_on_grid
from tests.conftest import binormal_dataset

#: Binormal shift solving AUC = 0.8 (A15 DGP 1).
D_08 = math.sqrt(2.0) * norm.ppf(0.8)


def true_binormal_roc(grid: np.ndarray, d: float) -> np.ndarray:
    """Population ROC of the equal-variance binormal DGP: R(t) = Phi(d + probit t)."""
    out = np.asarray(norm.cdf(d + norm.ppf(np.clip(grid, 1e-300, 1.0))))
    out[grid <= 0.0] = 0.0
    out[grid >= 1.0] = 1.0
    return out


def ks_reference_band(
    neg: np.ndarray, pos: np.ndarray, grid: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """A16 KS/DKW fixed-width reference band — the width yardstick."""
    alpha_m = 1.0 - math.sqrt(1.0 - alpha)  # Sidak across the two ECDFs
    c = math.sqrt(math.log(2.0 / alpha_m) / 2.0)
    d0 = c / math.sqrt(len(neg))
    d1 = c / math.sqrt(len(pos))
    upper = np.clip(
        empirical_roc_on_grid(neg, pos, np.clip(grid + d0, 0, 1)) + d1, 0, 1
    )
    lower = np.clip(
        empirical_roc_on_grid(neg, pos, np.clip(grid - d0, 0, 1)) - d1, 0, 1
    )
    lower[0] = 0.0
    upper[-1] = 1.0
    return lower, upper


class TestTrueRocContainment:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_band_contains_true_binormal_roc(self, seed):
        # the entire reason the band exists: at 95% simultaneous coverage the
        # true population ROC lies inside the band on typical draws (fixed
        # seeds; all were verified to be covered draws when recorded)
        y_true, y_score = binormal_dataset(150, 150, seed=seed)
        band = roc_band(y_true, y_score, n_boot=1500, random_state=seed)
        r = true_binormal_roc(band.fpr, D_08)
        assert (band.lower <= r + 1e-12).all(), "lower band crossed the true ROC"
        assert (r <= band.upper + 1e-12).all(), "upper band crossed the true ROC"


class TestKsYardstick:
    @pytest.mark.parametrize(("n", "seed"), [(100, 0), (100, 1), (400, 2)])
    def test_envelope_beats_ks_band_on_binormal(self, n, seed):
        # the paper's headline claim: the studentized envelope is strictly
        # tighter than the trivially valid KS/DKW band (measured ratios
        # 0.62-0.76 on these draws — assert the direction, not the margin)
        y_true, y_score = binormal_dataset(n, n, seed=seed)
        band = roc_band(y_true, y_score, n_boot=1500, random_state=seed)
        neg = np.sort(y_score[y_true == 0])
        pos = np.sort(y_score[y_true == 1])
        ks_lo, ks_hi = ks_reference_band(neg, pos, band.fpr, 0.05)
        assert band.band_area < float(np.mean(ks_hi - ks_lo))

    def test_envelope_beats_ks_band_on_heavy_tails(self):
        # distribution-free means the win cannot be a binormal artifact
        rng = np.random.default_rng(3)
        neg = rng.standard_t(3, 200)
        pos = rng.standard_t(3, 200) + 2.0
        y_true = np.r_[np.zeros(200, int), np.ones(200, int)]
        band = roc_band(y_true, np.r_[neg, pos], n_boot=1500, random_state=3)
        ks_lo, ks_hi = ks_reference_band(np.sort(neg), np.sort(pos), band.fpr, 0.05)
        assert band.band_area < float(np.mean(ks_hi - ks_lo))


class TestConfidenceOrdering:
    def _bands(self):
        y_true, y_score = binormal_dataset(120, 120, seed=5)
        return {
            c: roc_band(y_true, y_score, confidence=c, n_boot=1500, random_state=11)
            for c in (0.90, 0.95, 0.99)
        }

    def test_bands_nest_pointwise_in_confidence(self):
        # more confidence must never buy a narrower bound anywhere
        b = self._bands()
        assert (b[0.99].lower <= b[0.95].lower + 1e-12).all()
        assert (b[0.95].lower <= b[0.90].lower + 1e-12).all()
        assert (b[0.99].upper >= b[0.95].upper - 1e-12).all()
        assert (b[0.95].upper >= b[0.90].upper - 1e-12).all()

    def test_vacuous_region_grows_with_confidence(self):
        # exact theory: q_1 = 1 - (alpha / (2 j_max))^(1/n_neg) is strictly
        # decreasing in alpha, so the certifiable region shrinks as the
        # confidence level rises
        b = self._bands()
        assert b[0.90].vacuous_below < b[0.95].vacuous_below < b[0.99].vacuous_below


class TestSampleSizeScaling:
    def test_band_area_shrinks_at_root_n_rate(self):
        # quadrupling both classes must shrink the mean width, and roughly at
        # the sqrt(n) rate (window [1.5, 2.9] around the ideal factor 2;
        # measured 2.20 and 2.08 on these draws)
        areas = {}
        for n in (60, 240, 960):
            y_true, y_score = binormal_dataset(n, n, seed=7)
            areas[n] = roc_band(y_true, y_score, n_boot=1500, random_state=7).band_area
        assert areas[960] < areas[240] < areas[60]
        assert 1.5 < areas[60] / areas[240] < 2.9
        assert 1.5 < areas[240] / areas[960] < 2.9


def test_envelope_wider_than_wh_on_clean_binormal():
    """The distribution-free band pays a finite, bounded price for validity.

    On data where the parametric model is exactly right, Working-Hotelling
    is tighter (measured 0.095 vs 0.178 here) — the honest cost of dropping
    the binormal assumption. If the envelope ever undercuts the correctly
    specified parametric band on its home turf, something is over-tight and
    coverage is at risk.
    """
    y_true, y_score = binormal_dataset(300, 300, seed=9)
    env = roc_band(y_true, y_score, n_boot=1500, random_state=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RocciWarning)
        wh = roc_band(y_true, y_score, normal=True)
    assert wh.band_area < env.band_area
    assert env.band_area < 2.5 * wh.band_area, "validity should not cost 2.5x width"
