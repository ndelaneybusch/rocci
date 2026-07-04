"""Public API contract for ``roc_band`` / ``from_estimator``.

This suite exercises the orchestration layer (ingest -> validate -> kernel ->
assemble), so a pass certifies the wiring and the caller-facing contract, not
the statistics: the numerical correctness of the band *values* is proven in the
envelope, floors, statistics, and calibration suites. What you can rely on here
is that a well-formed call is turned into a structurally sound result and that a
malformed one fails loudly and specifically.

Guaranteed. Every returned ``RocBand`` is a valid ROC band — matching shapes,
values in [0, 1], ``lower <= upper``, both arms monotone non-decreasing, and the
band contains the empirical curve at every grid point (pinned endpoints
included). Provenance fields (``method``, ``backend``, ``n_neg``, ``n_pos``,
``n_boot``, ``random_state``) report what actually ran. ``auc`` equals the exact
Mann-Whitney statistic (ties weighted 1/2) to 1e-14 and ``auc_ci`` brackets it.
The warning/error taxonomy fires exactly at its declared thresholds and nowhere
else: a healthy call is silent; ``confidence < 0.90`` warns; ``n_boot < 100``
raises, ``[100, 1000)`` warns, ``>= 1000`` is silent; malformed
``confidence`` / ``random_state`` / ``n_threads`` / ``grid_size`` raise
``RocciError`` naming the offending argument. Results are frozen; ``.at()`` is
the right-continuous step of appendix A13 (NaN and out-of-range queries raise).
Exotic-but-legal inputs stay valid: float32 is bit-identical to its float64
widening, integer-rank scores reproduce the band (rank invariance), +/-inf is
handled, and the minimal ``n_neg = n_pos = 2`` problem yields the closed-form
vacuous region.

Limitations. Reproducibility is asserted only within a fixed backend and version
— Rust and NumPy agree statistically, not bit-for-bit; thread-invariance is
checked on Rust only. Coverage and width quality are out of scope for this file.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from rocci import RocBand, from_estimator, roc_band, show_versions
from rocci._exceptions import RocciError
from rocci._warnings import LowConfidenceWarning, RocciWarning, SmallSampleWarning
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

    def test_auc_is_exact_mann_whitney(self):
        y_true, y_score = binormal_dataset(100, 100, seed=3)
        band = roc_band(y_true, y_score, random_state=0)
        neg, pos = y_score[y_true == 0], y_score[y_true == 1]
        # literal pairwise oracle, ties weighted 1/2
        expected = (
            (pos[:, None] > neg[None, :]).sum()
            + 0.5 * (pos[:, None] == neg[None, :]).sum()
        ) / (len(neg) * len(pos))
        assert band.auc == pytest.approx(expected, abs=1e-14)
        assert band.auc_ci is not None
        lo, hi = band.auc_ci
        assert lo < band.auc < hi
        assert hi - lo < 0.25, "a 95% AUC CI at n=100 must be far tighter than this"

    def test_grid_size_override(self):
        y_true, y_score = binormal_dataset(100, 100, seed=4)
        band = roc_band(y_true, y_score, grid_size=17, random_state=0)
        assert band.fpr.shape == (17,)
        np.testing.assert_array_equal(band.fpr, np.linspace(0, 1, 17))

    def test_grid_size_two_is_all_pins(self):
        # the minimal legal grid is the two endpoints; both are pinned, so
        # the band is exactly the degenerate staircase
        y_true, y_score = binormal_dataset(60, 60, seed=4)
        band = roc_band(y_true, y_score, grid_size=2, random_state=0)
        assert band.fpr.tolist() == [0.0, 1.0]
        assert band.lower.tolist() == [0.0, 1.0]
        assert band.upper[-1] == 1.0
        assert band.attribution.tolist() == [3, 3]


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
    def test_clean_call_emits_no_rocci_warnings(self):
        # negative control for the whole taxonomy: a healthy call (n >= 20,
        # continuous scores, default n_boot and confidence) must be silent
        y_true, y_score = binormal_dataset(60, 60, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RocciWarning)
            roc_band(y_true, y_score, random_state=0)

    def test_low_confidence_warns(self):
        y_true, y_score = binormal_dataset(60, 60, seed=7)
        with pytest.warns(LowConfidenceWarning):
            roc_band(y_true, y_score, confidence=0.5, random_state=0)

    def test_confidence_boundary_at_090(self):
        # the declared threshold is confidence < 0.90 strictly
        y_true, y_score = binormal_dataset(60, 60, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("error", LowConfidenceWarning)
            roc_band(y_true, y_score, confidence=0.90, random_state=0)
        with pytest.warns(LowConfidenceWarning):
            roc_band(y_true, y_score, confidence=0.89, random_state=0)

    def test_n_boot_boundaries(self):
        # raise below 100, warn below 1000, silent at exactly 1000
        y_true, y_score = binormal_dataset(60, 60, seed=7)
        with pytest.raises(RocciError, match="n_boot"):
            roc_band(y_true, y_score, n_boot=99, random_state=0)
        with pytest.warns(RocciWarning, match="n_boot"):
            band = roc_band(y_true, y_score, n_boot=100, random_state=0)
        assert band.n_boot == 100  # exactly 100 is accepted (warn, not raise)
        with pytest.warns(RocciWarning, match="n_boot"):
            roc_band(y_true, y_score, n_boot=999, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RocciWarning)
            roc_band(y_true, y_score, n_boot=1000, random_state=0)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, np.nan])
    def test_bad_confidence_raises(self, confidence):
        y_true, y_score = binormal_dataset(60, 60, seed=8)
        with pytest.raises(RocciError):
            roc_band(y_true, y_score, confidence=confidence, random_state=0)

    def test_too_few_boot_raises(self):
        y_true, y_score = binormal_dataset(60, 60, seed=9)
        with pytest.raises(RocciError, match="n_boot"):
            roc_band(y_true, y_score, n_boot=50, random_state=0)

    def test_diagnostics_true_renders_figure(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        y_true, y_score = binormal_dataset(60, 60, seed=11)
        n_before = len(plt.get_fignums())
        roc_band(y_true, y_score, diagnostics=True, random_state=0)
        assert len(plt.get_fignums()) == n_before + 1
        plt.close("all")

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

    def test_at_between_grid_points_is_right_continuous_step(self):
        # a query strictly between grid points must return the value at the
        # grid point to its *left* (right-continuous step, appendix A13)
        band = self._band()
        mid = (band.fpr[:-1] + band.fpr[1:]) / 2.0
        lo, tp, up = band.at(mid)
        np.testing.assert_array_equal(lo, band.lower[:-1])
        np.testing.assert_array_equal(tp, band.tpr[:-1])
        np.testing.assert_array_equal(up, band.upper[:-1])

    def test_at_endpoints(self):
        band = self._band()
        lo, _tp, up = band.at([0.0, 1.0])
        assert lo[0] == band.lower[0] == 0.0
        assert up[1] == band.upper[-1] == 1.0

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


class TestInputRedTeam:
    """Exotic-but-legal inputs must produce exactly the float64 band."""

    def test_float32_scores_match_float64_cast(self):
        # ingestion promises a float64 pipeline: feeding float32 must be
        # bit-identical to feeding its exact float64 widening
        y_true, y_score = binormal_dataset(80, 80, seed=16)
        s32 = y_score.astype(np.float32)
        a = roc_band(y_true, s32, random_state=5)
        b = roc_band(y_true, s32.astype(np.float64), random_state=5)
        np.testing.assert_array_equal(a.lower, b.lower)
        np.testing.assert_array_equal(a.upper, b.upper)
        assert a.auc == b.auc

    def test_integer_rank_scores_reproduce_the_band(self):
        # rank invariance at the API level doubles as an integer-dtype check:
        # replacing scores by their integer ranks is a strictly monotone
        # transform, so the envelope band must be bit-identical
        y_true, y_score = binormal_dataset(80, 80, seed=17)
        ranks = np.argsort(np.argsort(y_score)).astype(np.int64)
        a = roc_band(y_true, y_score, random_state=6)
        b = roc_band(y_true, ranks, random_state=6)
        np.testing.assert_array_equal(a.lower, b.lower)
        np.testing.assert_array_equal(a.upper, b.upper)
        np.testing.assert_array_equal(a.attribution, b.attribution)

    def test_float16_scores_produce_valid_band(self):
        y_true, y_score = binormal_dataset(80, 80, seed=18)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RocciWarning)  # rounding may tie
            band = roc_band(y_true, y_score.astype(np.float16), random_state=7)
        assert (band.lower <= band.upper + 1e-12).all()
        assert (band.lower <= band.tpr + 1e-12).all()
        assert (band.tpr <= band.upper + 1e-12).all()

    def test_infinite_scores_end_to_end(self):
        # ±inf is documented as legal; the band must stay a valid ROC band
        # and keep containing the empirical curve
        y_true, y_score = binormal_dataset(60, 60, seed=19)
        y_score[0] = -np.inf
        y_score[1] = np.inf
        y_score[-1] = np.inf
        band = roc_band(y_true, y_score, random_state=8)
        assert np.isfinite(band.lower).all()
        assert np.isfinite(band.upper).all()
        assert (band.lower <= band.tpr + 1e-12).all()
        assert (band.tpr <= band.upper + 1e-12).all()
        assert not np.isnan(band.auc)

    def test_two_samples_per_class_is_the_documented_floor(self):
        # the smallest legal problem: n_neg = n_pos = 2. Grid collapses to
        # 3 points; q_1 = 1 - (alpha/50)^(1/2) in closed form, and every
        # interior grid point below it has a provably vacuous lower band.
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.warns(SmallSampleWarning):
            band = roc_band(y_true, y_score, random_state=9)
        assert band.fpr.tolist() == [0.0, 0.5, 1.0]
        assert band.vacuous_below == pytest.approx(1.0 - (0.05 / 50) ** 0.5, rel=1e-12)
        assert band.lower.tolist() == [0.0, 0.0, 1.0]  # 0.5 < q_1: vacuous
        assert band.upper[-1] == 1.0
        assert (band.lower <= band.upper).all()


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
