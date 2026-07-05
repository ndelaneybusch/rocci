"""Working-Hotelling path and normality diagnostics (``normal=True``).

The ``normal=True`` band trades distribution-freeness for tightness by assuming a
binormal model — a good deal exactly when that assumption holds and a silent
coverage failure when it does not. So the guarantee here has two halves: the band
is the correct parametric object, and its diagnostics make the assumption's
failure *visible* to the user rather than swallowing it.

Guaranteed. The band reproduces its closed forms — independently transcribed in
the oracle test — to float precision, its probit-space width scales as
``sqrt(chi2.ppf(1-alpha, 2))`` (right
df, not a z substitution), and on a deterministic quantile-grid "sample" it
contains the true binormal ROC and shrinks at the exact delta-method rate (width
halves when n quadruples). Provenance is distinct from the envelope path
(``method = "working_hotelling"``, ``n_boot`` / ``auc_ci`` / ``vacuous_below`` /
``random_state`` are ``None``, attribution all zero, a normality report attached),
and ignored arguments (``random_state``, low ``n_boot``) genuinely change
nothing. The diagnostics flag heavy-tailed data as suspect and stay silent on
clean binormal data, choose the per-class test correctly (Shapiro up to n=5000,
D'Agostino above; "insufficient" for constant/tiny classes, which never create
false suspicion), and append the heavy-ties clause when relevant. The contrast
test locks in the defining difference: the envelope is rank-invariant under a
sigmoid transform while the WH band moves.

Limitations. These tests certify that the WH machinery is correct and that its
assumption-violation *warning* fires — they do not certify coverage when the
binormal model is wrong (there is none to certify; that is the whole point of the
diagnostic). Asymptotic-rate claims use noise-free quantile grids, not random
draws.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.stats import chi2, norm

from rocci import roc_band
from rocci._warnings import NormalityWarning
from rocci.band.grids import empirical_roc_vertices
from rocci.band.normal import _probit_r2, normality_report, working_hotelling_band
from tests.conftest import binormal_dataset, binormal_scores


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


def quantile_scores(n: int, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 'perfect binormal sample': exact normal quantile grids.

    Sample moments converge to the population values as n grows with no
    Monte-Carlo noise, so asymptotic statements can be tested with tight,
    non-flaky tolerances.
    """
    q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return q, q + shift


def true_binormal_roc(grid: np.ndarray, d: float) -> np.ndarray:
    """Population ROC of the equal-variance binormal model, R(t) = Phi(d + probit t)."""
    out = np.asarray(norm.cdf(d + norm.ppf(np.clip(grid, 1e-300, 1.0))))
    out[grid <= 0.0] = 0.0
    out[grid >= 1.0] = 1.0
    return out


class TestWorkingHotellingExactness:
    def test_matches_closed_form_transcription_oracle(self):
        # independent transcription of the method-of-moments binormal closed
        # forms; any refactor of working_hotelling_band must reproduce them
        # to float precision
        neg, pos = binormal_scores(150, 130, seed=3)
        grid = np.linspace(0, 1, 65)
        alpha = 0.05
        n0, n1 = len(neg), len(pos)
        mu0, s0 = neg.mean(), neg.std(ddof=1)
        mu1, s1 = pos.mean(), pos.std(ddof=1)
        a = (mu1 - mu0) / s1
        b = s0 / s1
        var_a = 1.0 / n1 + b * b / n0 + a * a / (2.0 * n1)
        var_b = b * b / (2.0 * n0) + b * b / (2.0 * n1)
        cov_ab = a * b / (2.0 * n1)
        x = norm.ppf(np.clip(grid, 1e-9, 1.0 - 1e-9))
        se = np.sqrt(var_a + x * x * var_b + 2.0 * x * cov_ab)
        w = math.sqrt(chi2.ppf(1.0 - alpha, df=2))
        exp_lower = norm.cdf(a + b * x - w * se)
        exp_upper = norm.cdf(a + b * x + w * se)
        exp_lower[0] = 0.0
        exp_upper[-1] = 1.0

        lower, upper = working_hotelling_band(neg, pos, grid, alpha)
        np.testing.assert_allclose(lower, exp_lower, atol=1e-14)
        np.testing.assert_allclose(upper, exp_upper, atol=1e-14)

    def test_probit_width_scales_with_chi2_critical_value(self):
        # the band is symmetric around the fitted probit line with halfwidth
        # w(alpha) * se, so the probit-space width ratio between two levels is
        # exactly sqrt(chi2.ppf(0.99, 2) / chi2.ppf(0.95, 2)) — catching a
        # wrong df or a z-for-chi2 substitution that shape checks miss
        neg, pos = binormal_scores(200, 200, seed=4)
        grid = np.linspace(0, 1, 101)
        interior = (grid >= 0.05) & (grid <= 0.95)
        lo95, hi95 = working_hotelling_band(neg, pos, grid, 0.05)
        lo99, hi99 = working_hotelling_band(neg, pos, grid, 0.01)
        w95 = norm.ppf(hi95[interior]) - norm.ppf(lo95[interior])
        w99 = norm.ppf(hi99[interior]) - norm.ppf(lo99[interior])
        expected = math.sqrt(chi2.ppf(0.99, df=2) / chi2.ppf(0.95, df=2))
        np.testing.assert_allclose(w99 / w95, expected, rtol=1e-9)

    def test_asymptotic_consistency_on_true_binormal(self):
        # on a deterministic quantile-grid 'sample' of n = 50k the fitted band
        # must contain the true ROC and be tight: se ~ 1/sqrt(n) puts the
        # width below 0.018 everywhere (measured 0.0148)
        d = math.sqrt(2.0) * norm.ppf(0.8)
        neg, pos = quantile_scores(50_000, d)
        grid = np.linspace(0, 1, 201)
        lower, upper = working_hotelling_band(neg, pos, grid, 0.05)
        r = true_binormal_roc(grid, d)
        assert (lower <= r + 1e-12).all()
        assert (r <= upper + 1e-12).all()
        assert np.max(upper - lower) < 0.018

    def test_probit_width_halves_when_n_quadruples(self):
        # every variance term is proportional to 1/n at fixed (a, b), so
        # quadrupling n must halve the probit-space width — the delta-method
        # rate as an exact law, tight because quantile grids kill MC noise
        d = math.sqrt(2.0) * norm.ppf(0.8)
        grid = np.linspace(0, 1, 101)
        interior = (grid >= 0.1) & (grid <= 0.9)
        widths = {}
        for n in (12_500, 50_000):
            lo, hi = working_hotelling_band(*quantile_scores(n, d), grid, 0.05)
            widths[n] = norm.ppf(hi[interior]) - norm.ppf(lo[interior])
        np.testing.assert_allclose(widths[12_500] / widths[50_000], 2.0, rtol=1e-3)

    def test_constant_classes_yield_valid_band(self):
        # both classes constant: the std floor guards the fit; the band must
        # stay a valid (if vacuous) band, never NaN
        neg = np.zeros(50)
        pos = np.ones(50)
        grid = np.linspace(0, 1, 33)
        lower, upper = working_hotelling_band(neg, pos, grid, 0.05)
        assert np.isfinite(lower).all()
        assert np.isfinite(upper).all()
        assert (lower >= 0).all()
        assert (upper <= 1).all()
        assert (lower <= upper + 1e-12).all()
        assert lower[0] == 0.0
        assert upper[-1] == 1.0


class TestDiagnosticsEdgeCases:
    def test_insufficient_classes_never_create_suspicion(self):
        # classes too small (or constant) to test report ("insufficient",
        # nan, nan) and must not flag the fit: nan < 0.10 is False and a nan
        # R^2 is excluded from the suspect rule
        neg = np.array([0.0, 1.0])
        pos = np.array([2.0, 3.0])
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert report.neg_test == "insufficient"
        assert report.pos_test == "insufficient"
        assert math.isnan(report.neg_pvalue)
        assert math.isnan(report.probit_r2)
        assert report.suspect is False
        assert report.warning == ""

    def test_constant_class_reports_insufficient(self):
        neg = np.zeros(100)
        pos = np.linspace(0, 1, 100)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert report.neg_test == "insufficient"
        assert report.pos_test == "shapiro"

    def test_probit_r2_nan_below_ten_interior_vertices(self):
        neg, pos = binormal_scores(6, 6, seed=5)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        assert math.isnan(_probit_r2(fpr_v, tpr_v))

    def test_probit_r2_near_one_on_population_binormal_vertices(self):
        # the probit-probit ROC of a binormal pair is exactly linear, so the
        # population vertex list must fit with R^2 ~ 1
        d = math.sqrt(2.0) * norm.ppf(0.9)
        neg, pos = quantile_scores(2000, d)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        assert _probit_r2(fpr_v, tpr_v) > 0.9999

    @pytest.mark.parametrize(
        ("n", "expected"), [(5000, "shapiro"), (5001, "normaltest")]
    )
    def test_class_test_switches_at_shapiro_cap(self, n, expected):
        rng = np.random.default_rng(6)
        neg = rng.normal(0, 1, n)
        pos = rng.normal(1, 1, n)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert report.neg_test == expected
        assert report.pos_test == expected


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
