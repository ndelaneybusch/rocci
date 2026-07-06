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
clean binormal data, run each per-class check exactly on its validity window
(Shapiro-Francia for 5 <= n <= 5000, D'Agostino K² for n >= 20, all-nan for
constant/tiny classes, which never create false suspicion), OR-compose the
triggers so any single check can flag the fit, and append the heavy-ties clause
when relevant. A hypothesis sweep locks the report's internal consistency over
arbitrary score arrays. The contrast test locks in the defining difference: the
envelope is rank-invariant under a sigmoid transform while the WH band moves.

Limitations — tested as such. These tests certify that the WH machinery is
correct and that its assumption-violation *warning* fires — they do not certify
coverage when the binormal model is wrong (there is none to certify; that is
the whole point of the diagnostic). The silent-failure suite pins the honest
version of that statement: in mildly off-binormal regimes a meaningful fraction
of datasets pass every check while the WH band misses the true ROC far above
nominal, so a passing gate must never be read as a coverage certificate.
Asymptotic-rate claims use noise-free quantile grids, not random draws.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chi2, norm
from scipy.stats import t as student_t

from rocci import roc_band
from rocci._warnings import NormalityWarning
from rocci.band.grids import empirical_roc_vertices
from rocci.band.normal import (
    _SUSPECT_P,
    _SUSPECT_R2,
    _probit_r2,
    normality_report,
    working_hotelling_band,
)
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
        # classes too small (or constant) to check report all-nan and must
        # not flag the fit: nan < 0.10 is False and a nan R^2 is excluded
        # from the suspect rule
        neg = np.array([0.0, 1.0])
        pos = np.array([2.0, 3.0])
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert math.isnan(report.neg_sf_pvalue)
        assert math.isnan(report.neg_k2_pvalue)
        assert math.isnan(report.neg_skew)
        assert math.isnan(report.neg_pvalue)  # headline property: no checks
        assert math.isnan(report.probit_r2)
        assert report.suspect is False
        assert report.warning == ""

    def test_constant_class_reports_no_checks(self):
        neg = np.zeros(100)
        pos = np.linspace(0, 1, 100)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert math.isnan(report.neg_sf_pvalue)
        assert math.isnan(report.neg_k2_pvalue)
        assert not math.isnan(report.pos_sf_pvalue)
        assert not math.isnan(report.pos_k2_pvalue)

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
        ("n", "sf_runs", "k2_runs"),
        [
            (4, False, False),
            (10, True, False),
            (20, True, True),
            (5000, True, True),
            (5001, False, True),
        ],
    )
    def test_check_windows_by_class_size(self, n, sf_runs, k2_runs):
        # Shapiro-Francia runs on Royston's validated 5 <= n <= 5000 window,
        # D'Agostino K² from n = 20 up; outside a window the check is nan
        rng = np.random.default_rng(6)
        neg = rng.normal(0, 1, n)
        pos = rng.normal(1, 1, n)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert math.isnan(report.neg_sf_pvalue) != sf_runs
        assert math.isnan(report.pos_sf_pvalue) != sf_runs
        assert math.isnan(report.neg_k2_pvalue) != k2_runs
        assert math.isnan(report.pos_k2_pvalue) != k2_runs
        # moment effect sizes exist whenever the class is testable at all
        assert not math.isnan(report.neg_skew)
        assert not math.isnan(report.pos_excess_kurtosis)


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
        assert not np.isnan(band.normality.neg_sf_pvalue)

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
        # both checks apply at n = 500, and the effect sizes come along
        assert not np.isnan(report.pos_sf_pvalue)
        assert not np.isnan(report.pos_k2_pvalue)
        assert not np.isnan(report.pos_skew)
        assert not np.isnan(report.pos_excess_kurtosis)


class TestGateComposition:
    def test_one_bad_class_suffices_negative(self):
        # OR-composition: heavy-tailed negatives alone flag the fit even
        # though the positives are cleanly normal
        rng = np.random.default_rng(12)
        neg = rng.standard_t(3, 300)
        pos = rng.normal(2.0, 1.0, 300)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert report.suspect

    def test_one_bad_class_suffices_positive(self):
        rng = np.random.default_rng(13)
        neg = rng.normal(0.0, 1.0, 300)
        pos = rng.lognormal(0.8, 0.6, 300)
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert report.suspect

    def test_warning_reports_effect_sizes(self):
        # the warning is the artifact users act on: it must carry the moment
        # effect sizes, not just p-values
        rng = np.random.default_rng(14)
        neg = rng.standard_t(3, 400)
        pos = rng.standard_t(3, 400) + 2.0
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
        assert report.suspect
        assert "excess kurtosis" in report.warning
        assert "skew" in report.warning
        assert "SF p=" in report.warning
        assert "K2 p=" in report.warning


class TestReportInvariants:
    """Hypothesis sweep: the report is internally consistent on any input."""

    scores = st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        min_size=2,
        max_size=120,
    ).map(lambda v: np.asarray(v, dtype=np.float64))

    @settings(max_examples=80, deadline=None)
    @given(neg=scores, pos=scores)
    def test_report_consistency(self, neg, pos):
        fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
        report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)

        # suspect is exactly the OR of the published triggers (nan never fires)
        pvalues = (
            report.neg_sf_pvalue,
            report.neg_k2_pvalue,
            report.pos_sf_pvalue,
            report.pos_k2_pvalue,
        )
        expected = any(p < _SUSPECT_P for p in pvalues) or (
            not math.isnan(report.probit_r2) and report.probit_r2 < _SUSPECT_R2
        )
        assert report.suspect == expected
        assert (report.warning != "") == report.suspect

        # every p-value is nan or a probability
        for p in pvalues:
            assert math.isnan(p) or 0.0 <= p <= 1.0

        # check windows: SF on [5, 5000], K² from 20, nothing on a
        # (numerically) constant class
        for x, sf_p, k2_p in (
            (neg, report.neg_sf_pvalue, report.neg_k2_pvalue),
            (pos, report.pos_sf_pvalue, report.pos_k2_pvalue),
        ):
            testable = len(x) >= 3 and np.ptp(x) > 0 and np.std(x) > 0
            assert math.isnan(sf_p) == (not (testable and 5 <= len(x) <= 5000))
            assert math.isnan(k2_p) == (not (testable and len(x) >= 20))

        # the headline property is the smallest applicable check p-value
        valid = [
            p for p in (report.neg_sf_pvalue, report.neg_k2_pvalue) if not math.isnan(p)
        ]
        if valid:
            assert report.neg_pvalue == min(valid)
        else:
            assert math.isnan(report.neg_pvalue)


def _true_roc_t_shift(grid: np.ndarray, df: float, delta: float) -> np.ndarray:
    """Population ROC of a location-shifted Student-t pair."""
    g = np.clip(grid, 1e-12, 1.0 - 1e-12)
    return 1.0 - student_t.cdf(student_t.ppf(1.0 - g, df) - delta, df)


def _true_roc_bimodal_neg(grid: np.ndarray, d: float, mu: float) -> np.ndarray:
    """Population ROC with symmetric-bimodal negatives and N(mu, 1) positives."""
    c = np.linspace(-10.0, 10.0 + mu, 8001)
    fpr = 1.0 - 0.5 * (norm.cdf(c - d) + norm.cdf(c + d))
    tpr = 1.0 - norm.cdf(c - mu)
    return np.interp(grid, fpr[::-1], tpr[::-1])


@pytest.mark.slow
class TestSilentFailures:
    """The gate is a tripwire, not a certificate — and these tests keep it so.

    In mildly off-binormal regimes a meaningful fraction of datasets pass
    every check while the WH band misses the true ROC far above the nominal
    5%. These tests pin that down: if a future change makes them fail by
    eliminating the silent failures, the "no safe diagnostic region" warning
    language should be revisited; if it makes the pass rate collapse to ~0,
    the gate has become so trigger-happy the WH path is effectively dead.
    Thresholds sit at a third to a half of the measured values (pass
    ~0.14/0.22, miss-given-pass ~0.39/0.21 at these seeds) so they fail on
    real regime change, not Monte-Carlo wiggle.
    """

    GRID = np.linspace(0.01, 0.99, 99)

    def _sweep(self, sampler, true_roc, n_rep=400, seed=42):
        rng = np.random.default_rng(seed)
        n_pass = miss_given_pass = 0
        for _ in range(n_rep):
            neg, pos = sampler(rng)
            fpr_v, tpr_v = empirical_roc_vertices(neg, pos)
            report = normality_report(neg, pos, fpr_v, tpr_v, heavy_ties=False)
            if report.suspect:
                continue
            n_pass += 1
            lower, upper = working_hotelling_band(neg, pos, self.GRID, 0.05)
            truth = true_roc(self.GRID)
            covered = ((lower <= truth + 1e-12) & (truth <= upper + 1e-12)).all()
            miss_given_pass += not covered
        return n_pass / n_rep, miss_given_pass / max(n_pass, 1)

    def test_mild_heavy_tails_slip_through(self):
        # t(df=10) classes at n=150: tails too mild for the checks to see
        # reliably, heavy enough to break WH coverage
        df, n, delta = 10, 150, 1.5
        pass_rate, miss_rate = self._sweep(
            lambda rng: (rng.standard_t(df, n), rng.standard_t(df, n) + delta),
            lambda grid: _true_roc_t_shift(grid, df, delta),
        )
        assert pass_rate > 0.05, "gate now rejects nearly everything here"
        assert miss_rate > 0.15, "silent failures gone - revisit warning language"

    def test_mild_bimodality_slips_through(self):
        # weakly separated bimodal negatives at n=300: kurtosis barely moves,
        # but the binormal fit converges to the wrong curve
        d, n, mu = 1.0, 300, 2.0

        def sampler(rng):
            sign = np.where(rng.integers(0, 2, n) == 1, d, -d)
            return rng.normal(0.0, 1.0, n) + sign, rng.normal(mu, 1.0, n)

        pass_rate, miss_rate = self._sweep(
            sampler, lambda grid: _true_roc_bimodal_neg(grid, d, mu)
        )
        assert pass_rate > 0.08, "gate now rejects nearly everything here"
        assert miss_rate > 0.10, "silent failures gone - revisit warning language"

    def test_gross_departures_are_caught(self):
        # the tripwire half of the story: blatant heavy tails at real sample
        # sizes must fire essentially always
        df, n, delta = 3, 500, 2.0
        pass_rate, _ = self._sweep(
            lambda rng: (rng.standard_t(df, n), rng.standard_t(df, n) + delta),
            lambda grid: _true_roc_t_shift(grid, df, delta),
            n_rep=200,
        )
        assert pass_rate < 0.02


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
