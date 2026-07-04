"""Public API contract for ``roc_band`` / ``from_estimator``.

Risk mitigated: the orchestration layer wires ingestion, validation, the
bootstrap kernel, and result assembly together — a break here means the
validated statistics are correct but the band a user receives is not. Tests
assert the band shape/ordering invariants, the warning-and-error taxonomy fires
exactly when specified, result immutability, and reproducibility.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rocci import RocBand, from_estimator, roc_band, show_versions
from rocci._exceptions import RocciError
from rocci._warnings import LowConfidenceWarning
from rocci.backend import BACKEND
from tests.conftest import binormal_dataset


class TestBandStructure:
    def test_returns_rocband_with_provenance(self):
        y_true, y_score = binormal_dataset(80, 80, seed=1)
        band = roc_band(y_true, y_score, random_state=0)
        assert isinstance(band, RocBand)
        assert band.method == "envelope"
        assert band.backend == BACKEND
        assert band.n_neg == 80
        assert band.n_pos == 80
        assert band.n_boot == 2000
        assert band.random_state == 0
        assert band.normality is None

    def test_band_is_valid_roc_band(self):
        y_true, y_score = binormal_dataset(120, 90, seed=2)
        band = roc_band(y_true, y_score, random_state=0)
        assert band.fpr.shape == band.lower.shape == band.upper.shape
        assert (band.lower >= 0.0).all()
        assert (band.upper <= 1.0).all()
        assert (band.lower <= band.upper + 1e-12).all()
        # contains the empirical curve everywhere (pinned endpoints included)
        assert (band.lower <= band.tpr + 1e-12).all()
        assert (band.tpr <= band.upper + 1e-12).all()
        # monotone non-decreasing arms
        assert (np.diff(band.lower) >= -1e-12).all()
        assert (np.diff(band.upper) >= -1e-12).all()

    def test_auc_and_ci_present_and_ordered(self):
        y_true, y_score = binormal_dataset(100, 100, seed=3)
        band = roc_band(y_true, y_score, random_state=0)
        assert 0.0 <= band.auc <= 1.0
        assert band.auc_ci is not None
        lo, hi = band.auc_ci
        assert lo <= hi

    def test_grid_size_override(self):
        y_true, y_score = binormal_dataset(100, 100, seed=4)
        band = roc_band(y_true, y_score, grid_size=17, random_state=0)
        assert band.fpr.shape == (17,)


class TestReproducibility:
    def test_same_seed_same_band(self):
        y_true, y_score = binormal_dataset(80, 80, seed=5)
        a = roc_band(y_true, y_score, random_state=7)
        b = roc_band(y_true, y_score, random_state=7)
        np.testing.assert_array_equal(a.lower, b.lower)
        np.testing.assert_array_equal(a.upper, b.upper)

    @pytest.mark.skipif(BACKEND != "rust", reason="threads only affect the Rust core")
    def test_thread_count_does_not_change_band(self):
        y_true, y_score = binormal_dataset(80, 80, seed=6)
        a = roc_band(y_true, y_score, random_state=1, n_threads=1)
        b = roc_band(y_true, y_score, random_state=1, n_threads=4)
        np.testing.assert_array_equal(a.lower, b.lower)


class TestWarningsAndErrors:
    def test_low_confidence_warns(self):
        y_true, y_score = binormal_dataset(60, 60, seed=7)
        with pytest.warns(LowConfidenceWarning):
            roc_band(y_true, y_score, confidence=0.5, random_state=0)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, np.nan])
    def test_bad_confidence_raises(self, confidence):
        y_true, y_score = binormal_dataset(60, 60, seed=8)
        with pytest.raises(RocciError):
            roc_band(y_true, y_score, confidence=confidence, random_state=0)

    def test_too_few_boot_raises(self):
        y_true, y_score = binormal_dataset(60, 60, seed=9)
        with pytest.raises(RocciError, match="n_boot"):
            roc_band(y_true, y_score, n_boot=50, random_state=0)

    def test_diagnostics_true_deferred(self):
        y_true, y_score = binormal_dataset(60, 60, seed=11)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            roc_band(y_true, y_score, diagnostics=True)

    @pytest.mark.parametrize("random_state", [-1, 2**64, "seven", 1.5])
    def test_bad_random_state_raises_rocci_error(self, random_state):
        y_true, y_score = binormal_dataset(60, 60, seed=10)
        with pytest.raises(RocciError, match="random_state"):
            roc_band(y_true, y_score, random_state=random_state)

    def test_n_threads_minus_one_means_all_cores(self):
        # sklearn n_jobs muscle memory: -1 must behave like None
        y_true, y_score = binormal_dataset(60, 60, seed=10)
        a = roc_band(y_true, y_score, random_state=3, n_threads=-1)
        b = roc_band(y_true, y_score, random_state=3, n_threads=None)
        np.testing.assert_array_equal(a.lower, b.lower)

    @pytest.mark.parametrize("n_threads", [-2, "four"])
    def test_bad_n_threads_raises_rocci_error(self, n_threads):
        y_true, y_score = binormal_dataset(60, 60, seed=10)
        with pytest.raises(RocciError, match="n_threads"):
            roc_band(y_true, y_score, random_state=0, n_threads=n_threads)

    @pytest.mark.parametrize("grid_size", [0, 1, -3])
    def test_bad_grid_size_raises_rocci_error(self, grid_size):
        y_true, y_score = binormal_dataset(60, 60, seed=10)
        with pytest.raises(RocciError, match="grid_size"):
            roc_band(y_true, y_score, grid_size=grid_size, random_state=0)


class TestResultMethods:
    def _band(self, seed=12):
        y_true, y_score = binormal_dataset(80, 80, seed=seed)
        return roc_band(y_true, y_score, random_state=0)

    def test_immutable(self):
        band = self._band()
        with pytest.raises(dataclasses.FrozenInstanceError):
            band.lower = np.zeros_like(band.lower)

    def test_at_matches_band_at_grid_points(self):
        band = self._band()
        # querying exact grid points returns the stored values (step convention)
        lo, tp, up = band.at(band.fpr)
        np.testing.assert_array_equal(lo, band.lower)
        np.testing.assert_array_equal(tp, band.tpr)
        np.testing.assert_array_equal(up, band.upper)

    def test_at_out_of_range_raises(self):
        band = self._band()
        with pytest.raises(RocciError, match=r"\[0, 1\]"):
            band.at([1.5])

    def test_at_nan_raises(self):
        # NaN passes silent min/max comparisons; it must not return garbage
        band = self._band()
        with pytest.raises(RocciError, match="NaN"):
            band.at([0.5, np.nan])

    def test_band_area_matches_mean_width(self):
        band = self._band()
        assert band.band_area == pytest.approx(np.mean(band.upper - band.lower))

    def test_summary_reports_key_facts(self):
        band = self._band()
        text = band.summary()
        assert "n_neg=80" in text
        assert "AUC" in text
        assert "envelope" in text

    def test_to_dataframe_columns(self):
        band = self._band()
        df = band.to_dataframe()
        assert list(df.columns) == ["fpr", "lower", "tpr", "upper", "attribution"]
        assert len(df) == band.fpr.shape[0]


class TestFromEstimator:
    def test_predict_proba_path(self):
        y_true, raw = binormal_dataset(80, 80, seed=13)
        p1 = 1.0 / (1.0 + np.exp(-raw))

        class ProbaClf:
            def predict_proba(self, x):
                x = np.asarray(x)
                return np.column_stack([1.0 - x.ravel(), x.ravel()])

        band = from_estimator(ProbaClf(), p1[:, None], y_true, random_state=0)
        assert band.method == "envelope"
        assert band.n_pos == 80

    def test_decision_function_path(self):
        y_true, raw = binormal_dataset(80, 80, seed=14)

        class MarginClf:
            def decision_function(self, x):
                return np.asarray(x).ravel()

        band = from_estimator(MarginClf(), raw[:, None], y_true, random_state=0)
        assert band.n_neg == 80

    def test_missing_response_method_raises(self):
        class Empty:
            pass

        y_true, raw = binormal_dataset(30, 30, seed=15)
        with pytest.raises(RocciError, match="predict_proba"):
            from_estimator(Empty(), raw[:, None], y_true)

    def test_invalid_response_method_raises(self):
        class MarginClf:
            def decision_function(self, x):
                return np.asarray(x).ravel()

        y_true, raw = binormal_dataset(30, 30, seed=15)
        with pytest.raises(RocciError, match="response_method"):
            from_estimator(
                MarginClf(),
                raw[:, None],
                y_true,
                response_method="proba",  # ty: ignore[invalid-argument-type]
            )


def test_show_versions_reports_backend(capsys):
    show_versions()
    out = capsys.readouterr().out
    assert "rocci:" in out
    assert BACKEND in out
    assert "numpy:" in out
