"""rocci.special vs scipy: equivalence on the supported domains.

These implementations exist so the band core does not need a general
special-functions library; correctness is therefore defined as agreement with
scipy to near machine precision. The equivalence sweeps below are the primary
oracle. The handful of non-scipy tests cover only what equivalence cannot:
edge-value conventions, input validation, and the accuracy of ``beta_ppf`` at
quantiles far smaller than one ulp of 1 (where absolute comparisons against
scipy say nothing about relative error).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta
from scipy.stats import chi2 as scipy_chi2
from scipy.stats import norm as scipy_norm
from scipy.stats import normaltest as scipy_normaltest

from rocci.special import beta_ppf, chi2_ppf, dagostino_k2, ndtr, ndtri


class TestNdtri:
    def test_matches_scipy_across_the_unit_interval(self):
        p = np.linspace(1e-12, 1.0 - 1e-12, 20001)
        np.testing.assert_allclose(ndtri(p), scipy_norm.ppf(p), rtol=1e-13, atol=0.0)

    def test_matches_scipy_in_the_far_tails(self):
        p = 10.0 ** np.arange(-300.0, -1.0)
        np.testing.assert_allclose(ndtri(p), scipy_norm.ppf(p), rtol=1e-13, atol=0.0)
        # upper tail: symmetric region reachable in double precision
        p_hi = 1.0 - 10.0 ** np.arange(-15.0, -1.0)
        np.testing.assert_allclose(
            ndtri(p_hi), scipy_norm.ppf(p_hi), rtol=1e-13, atol=0.0
        )

    def test_edge_and_invalid_values_match_scipy_conventions(self):
        out = ndtri(np.array([0.0, 1.0, -0.5, 1.5, np.nan]))
        assert out[0] == -np.inf
        assert out[1] == np.inf
        assert np.isnan(out[2:]).all()

    def test_scalar_input_returns_python_float(self):
        z = ndtri(0.975)
        assert isinstance(z, float)
        assert z == pytest.approx(1.959963984540054, rel=1e-14)

    def test_round_trip_with_ndtr(self):
        p = np.linspace(1e-8, 1.0 - 1e-8, 1001)
        np.testing.assert_allclose(ndtr(ndtri(p)), p, rtol=1e-12, atol=0.0)


class TestNdtr:
    def test_matches_scipy(self):
        # Tight agreement over the range the band code evaluates (probit
        # values are clipped to |x| <~ 6 upstream), looser in the deep tail
        # where the C-library and Cephes erfc roundings legitimately drift a
        # few hundred ulps apart near 1e-300.
        x = np.linspace(-10.0, 8.0, 20001)
        np.testing.assert_allclose(ndtr(x), scipy_norm.cdf(x), rtol=1e-13, atol=0.0)
        deep = np.linspace(-37.0, -10.0, 5001)
        np.testing.assert_allclose(
            ndtr(deep), scipy_norm.cdf(deep), rtol=1e-12, atol=0.0
        )

    def test_scalar_and_shape_handling(self):
        assert ndtr(0.0) == 0.5
        assert ndtr(np.zeros((2, 3))).shape == (2, 3)


class TestChi2Ppf:
    def test_matches_scipy_at_df2(self):
        q = np.linspace(0.0, 1.0 - 1e-12, 10001)
        ours = np.array([chi2_ppf(float(v), df=2) for v in q])
        np.testing.assert_allclose(ours, scipy_chi2.ppf(q, df=2), rtol=1e-13, atol=0.0)
        assert chi2_ppf(1.0, df=2) == math.inf

    def test_other_df_is_rejected(self):
        with pytest.raises(NotImplementedError, match="df=2"):
            chi2_ppf(0.95, df=3)


class TestBetaPpf:
    # Shapes as the Beta floor uses them: Beta(j, n + 1 - j) for the j-th
    # order statistic of n uniforms, at extreme upper levels.
    @pytest.mark.parametrize("n", [5, 25, 100, 10_000, 1_000_000])
    @pytest.mark.parametrize("q", [0.05, 0.5, 0.9, 1.0 - 0.05 / 50, 0.999])
    def test_order_statistic_shapes_match_scipy(self, n, q):
        js = np.arange(1, min(25, n) + 1)
        ours = beta_ppf(q, js, n + 1 - js)
        np.testing.assert_allclose(
            ours, scipy_beta.ppf(q, js, n + 1 - js), rtol=1e-10, atol=0.0
        )

    def test_general_integer_shapes_match_scipy(self):
        for a, b in [(1, 1), (3, 7), (40, 2), (17, 17)]:
            for q in [0.001, 0.3, 0.7, 0.999]:
                assert beta_ppf(q, a, b) == pytest.approx(
                    scipy_beta.ppf(q, a, b), rel=1e-10
                )

    def test_tiny_quantiles_match_a_closed_form_oracle(self):
        # Beta(1, n) has CDF 1 - (1 - x)**n, so its ppf has an exact closed
        # form independent of scipy — an oracle for relative accuracy at a
        # quantile (~7e-6 here) far below the scale where bisection on [0, 1]
        # could hide absolute error.
        n = 1_000_000
        q = 0.999
        exact = -math.expm1(math.log1p(-q) / n)
        assert beta_ppf(q, 1, n) == pytest.approx(exact, rel=1e-12)

    def test_edges_and_validation(self):
        assert beta_ppf(0.0, 3, 4) == 0.0
        assert beta_ppf(1.0, 3, 4) == 1.0
        assert math.isnan(beta_ppf(1.5, 3, 4))
        with pytest.raises(ValueError, match="integer"):
            beta_ppf(0.5, np.array([1.5]), np.array([2.5]))
        with pytest.raises(ValueError, match="positive"):
            beta_ppf(0.5, 0, 4)

    def test_scalar_shapes_return_python_float(self):
        assert isinstance(beta_ppf(0.5, 2, 3), float)

    def test_monotone_in_q(self):
        qs = np.linspace(0.001, 0.999, 200)
        vals = np.array([beta_ppf(float(q), 5, 96) for q in qs])
        assert (np.diff(vals) > 0).all()


class TestDagostinoK2:
    @pytest.mark.parametrize(
        "sample",
        [
            lambda rng: rng.normal(size=6000),
            lambda rng: rng.lognormal(size=6000),  # right-skewed
            lambda rng: rng.standard_t(df=3, size=6000),  # heavy-tailed
            lambda rng: rng.uniform(size=6000),  # light-tailed
            lambda rng: rng.normal(size=20),  # smallest supported n
        ],
    )
    def test_matches_scipy_normaltest(self, sample):
        x = sample(np.random.default_rng(42))
        stat, p = dagostino_k2(x)
        expected = scipy_normaltest(x)
        assert stat == pytest.approx(float(expected.statistic), rel=1e-12)
        assert p == pytest.approx(float(expected.pvalue), rel=1e-12)

    def test_rejects_unsupported_samples(self):
        with pytest.raises(ValueError, match="n >= 20"):
            dagostino_k2(np.arange(19, dtype=np.float64))
        with pytest.raises(ValueError, match="non-constant"):
            dagostino_k2(np.ones(100))
