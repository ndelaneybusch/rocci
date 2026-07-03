"""Working-Hotelling path and normality diagnostics (``normal=True``).

Risk mitigated: the parametric band is only safe to offer if its failure modes
are visible. These tests pin the WH result's provenance and shape, confirm the
diagnostics flag non-binormal data (and stay quiet on clean binormal data),
and lock in the rank-invariance contrast that distinguishes the WH band from
the distribution-free envelope.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rocci import roc_band
from rocci._warnings import NormalityWarning
from tests.conftest import binormal_dataset


def heavy_tailed_dataset(
    n_neg: int, n_pos: int, seed: int, shift: float = 2.0, tie_step: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_score)`` with Student-t (df=3) class scores.

    The heavy tails make the binormal fit untenable; ``tie_step`` additionally
    rounds the scores to that granularity for the heavy-ties regime.
    """
    rng = np.random.default_rng(seed)
    neg = rng.standard_t(3, n_neg)
    pos = rng.standard_t(3, n_pos) + shift
    if tie_step is not None:
        neg = np.round(neg / tie_step) * tie_step
        pos = np.round(pos / tie_step) * tie_step
    y_true = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    y_score = np.concatenate([neg, pos])
    return y_true, y_score


class TestBandStructure:
    def test_provenance_fields(self):
        y_true, y_score = binormal_dataset(300, 300, seed=2)
        band = roc_band(y_true, y_score, normal=True)
        assert band.method == "working_hotelling"
        assert band.n_boot is None
        assert band.auc_ci is None
        assert band.vacuous_below is None
        assert band.random_state is None
        assert (band.attribution == 0).all()
        assert band.normality is not None

    def test_band_within_unit_square(self):
        y_true, y_score = binormal_dataset(200, 200, seed=2)
        band = roc_band(y_true, y_score, normal=True)
        assert band.fpr.shape == band.lower.shape == band.upper.shape
        assert (band.lower >= 0.0).all()
        assert (band.upper <= 1.0).all()
        assert (band.lower <= band.upper + 1e-12).all()
        # endpoints are pinned even though the WH band is not forced monotone
        assert band.lower[0] == 0.0
        assert band.upper[-1] == 1.0


class TestIgnoredArguments:
    def test_random_state_does_not_change_band(self):
        y_true, y_score = binormal_dataset(200, 200, seed=3)
        a = roc_band(y_true, y_score, normal=True, random_state=0)
        b = roc_band(y_true, y_score, normal=True, random_state=999)
        np.testing.assert_array_equal(a.lower, b.lower)
        np.testing.assert_array_equal(a.upper, b.upper)

    def test_low_n_boot_does_not_raise(self):
        y_true, y_score = binormal_dataset(200, 200, seed=4)
        # n_boot is ignored on the WH path, so the usual >= 100 check is skipped
        band = roc_band(y_true, y_score, normal=True, n_boot=50)
        assert band.method == "working_hotelling"


class TestNormalityDiagnostics:
    def test_warns_on_heavy_tailed_data(self):
        y_true, y_score = heavy_tailed_dataset(400, 400, seed=5)
        with pytest.warns(NormalityWarning):
            band = roc_band(y_true, y_score, normal=True)
        assert band.normality is not None
        assert band.normality.suspect

    def test_quiet_on_clean_binormal(self):
        y_true, y_score = binormal_dataset(500, 500, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", NormalityWarning)
            band = roc_band(y_true, y_score, normal=True)
        assert band.normality is not None
        assert band.normality.suspect is False
        assert band.normality.warning == ""
        assert band.normality.neg_test == "shapiro"

    def test_ties_clause_appended(self):
        y_true, y_score = heavy_tailed_dataset(400, 400, seed=7, tie_step=0.5)
        with pytest.warns(NormalityWarning) as record:
            roc_band(y_true, y_score, normal=True)
        # ingestion also emits a TiesWarning; pick the normality message
        message = next(
            str(w.message) for w in record if issubclass(w.category, NormalityWarning)
        )
        assert "heavy ties" in message

    def test_report_attached_even_when_clean(self):
        y_true, y_score = binormal_dataset(500, 500, seed=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NormalityWarning)
            band = roc_band(y_true, y_score, normal=True)
        report = band.normality
        assert report is not None
        assert not np.isnan(report.probit_r2)
        assert report.pos_test in ("shapiro", "normaltest")


class TestRankInvarianceContrast:
    def test_envelope_invariant_but_wh_changes(self):
        y_true, y_score = binormal_dataset(200, 200, seed=9)
        sigmoid = 1.0 / (1.0 + np.exp(-y_score))

        env_raw = roc_band(y_true, y_score, random_state=0)
        env_sig = roc_band(y_true, sigmoid, random_state=0)
        # the envelope is rank-based: a monotone transform leaves it unchanged
        np.testing.assert_array_equal(env_raw.lower, env_sig.lower)
        np.testing.assert_array_equal(env_raw.upper, env_sig.upper)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NormalityWarning)
            wh_raw = roc_band(y_true, y_score, normal=True)
            wh_sig = roc_band(y_true, sigmoid, normal=True)
        # the WH band depends on the raw score scale, so it must move
        assert not np.allclose(wh_raw.upper, wh_sig.upper)


class TestSummary:
    def test_summary_labels_working_hotelling(self):
        y_true, y_score = binormal_dataset(300, 300, seed=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NormalityWarning)
            text = roc_band(y_true, y_score, normal=True).summary()
        assert "Working-Hotelling" in text

    def test_summary_reports_suspect(self):
        y_true, y_score = heavy_tailed_dataset(400, 400, seed=11)
        with pytest.warns(NormalityWarning):
            text = roc_band(y_true, y_score, normal=True).summary()
        assert "SUSPECT" in text
