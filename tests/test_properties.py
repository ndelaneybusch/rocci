"""Property-based invariants of the assembled confidence band.

The example-based suites prove behavior on inputs someone chose; this one uses
Hypothesis to attack the same invariants with inputs no one chose — random sizes,
shifts, and four distribution families (normal, uniform, lognormal, heavy-tie
rounded) — so a band that is only accidentally valid on the curated fixtures gets
found out.

Guaranteed. Across the generated inputs the assembled band always holds its
structural invariants: values in [0, 1], ``lower <= upper``, both arms monotone,
endpoints pinned with the correct attribution, and a lower band that is exactly
zero throughout the vacuous region below ``q_1``. The envelope is bit-identical
under exact strictly-monotone rescaling (dyadic scale factors, which are exact in
floating point) — the rank invariance that certifies the band reads only the
order of the scores, not their magnitudes. The retained envelope arm never
escapes the pointwise min/max of the full bootstrap matrix, and a wide
``alpha = 0.5`` still yields a valid band.

Limitations. Property tests certify *invariants*, not values or coverage — they
say the band is always well-formed and rank-invariant, not that it is the
statistically correct band (golden-master and calibration cover that). Search is
bounded (``max_examples`` 10-25); it samples the input space, it does not
exhaust it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rocci import roc_band
from rocci.backend._fallback import bootstrap_tpr_matrix_numpy
from rocci.band.envelope import ATTR_PINNED, assemble_envelope_band
from rocci.band.floors import beta_floor_vacuous_below
from rocci.band.grids import grid_k_indices, make_grid


@st.composite
def score_samples(draw):
    n_neg = draw(st.integers(min_value=5, max_value=50))
    n_pos = draw(st.integers(min_value=5, max_value=50))
    seed = draw(st.integers(min_value=0, max_value=2**31))
    dist = draw(st.sampled_from(["normal", "uniform", "lognormal", "ties"]))
    rng = np.random.default_rng(seed)
    shift = draw(st.floats(min_value=0.0, max_value=3.0))
    if dist == "normal":
        neg, pos = rng.normal(0, 1, n_neg), rng.normal(shift, 1, n_pos)
    elif dist == "uniform":
        neg, pos = rng.uniform(0, 1, n_neg), rng.uniform(0, 1, n_pos) + shift / 3
    elif dist == "lognormal":
        neg, pos = rng.lognormal(0, 1, n_neg), rng.lognormal(shift / 3, 1, n_pos)
    else:
        neg = np.round(rng.normal(0, 1, n_neg), 1)
        pos = np.round(rng.normal(shift, 1, n_pos), 1)
    return np.sort(neg), np.sort(pos), seed


@given(score_samples(), st.sampled_from([0.05, 0.1]))
@settings(max_examples=25, deadline=None)
def test_band_invariants(sample, alpha):
    neg, pos, seed = sample
    grid = make_grid(len(neg))
    boot = bootstrap_tpr_matrix_numpy(
        neg, pos, grid_k_indices(grid, len(neg)), n_boot=150, seed=seed
    )
    band = assemble_envelope_band(boot, grid, neg, pos, alpha)

    assert (band.lower >= 0).all()
    assert (band.upper <= 1).all()
    assert (band.lower <= band.upper + 1e-12).all()
    assert (np.diff(band.lower) >= -1e-12).all(), "lower band not monotone"
    assert (np.diff(band.upper) >= -1e-12).all(), "upper band not monotone"
    assert band.lower[0] == 0.0
    assert band.upper[-1] == 1.0
    assert band.attribution[0] == ATTR_PINNED

    q1 = beta_floor_vacuous_below(len(neg), alpha)
    in_vacuous = (grid > 0) & (grid < q1)
    assert (band.lower[in_vacuous] == 0.0).all(), "vacuous region must be 0"


@given(
    st.integers(min_value=0, max_value=2**31), st.sampled_from([0.5, 2.0, 8.0, 1024.0])
)
@settings(max_examples=10, deadline=None)
def test_envelope_band_is_rank_invariant(seed, scale):
    # dyadic scaling is an exact strictly-monotone transform in floating
    # point, so the band must be bit-identical at the same seed
    rng = np.random.default_rng(seed)
    neg = np.sort(rng.normal(0, 1, 40))
    pos = np.sort(rng.normal(1, 1, 40))
    grid = make_grid(40)
    k = grid_k_indices(grid, 40)

    boot_a = bootstrap_tpr_matrix_numpy(neg, pos, k, n_boot=200, seed=seed)
    band_a = assemble_envelope_band(boot_a, grid, neg, pos, 0.05)
    boot_b = bootstrap_tpr_matrix_numpy(neg * scale, pos * scale, k, 200, seed=seed)
    band_b = assemble_envelope_band(boot_b, grid, neg * scale, pos * scale, 0.05)

    np.testing.assert_array_equal(band_a.lower, band_b.lower)
    np.testing.assert_array_equal(band_a.upper, band_b.upper)
    np.testing.assert_array_equal(band_a.attribution, band_b.attribution)


def test_wide_alpha_rejects_nothing_weird():
    # alpha=0.5: retention keeps only half the curves; band still valid
    rng = np.random.default_rng(0)
    neg = np.sort(rng.normal(0, 1, 30))
    pos = np.sort(rng.normal(1, 1, 30))
    grid = make_grid(30)
    boot = bootstrap_tpr_matrix_numpy(neg, pos, grid_k_indices(grid, 30), 100, seed=1)
    band = assemble_envelope_band(boot, grid, neg, pos, 0.5)
    assert (band.lower <= band.upper + 1e-12).all()


@given(score_samples(), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=25, deadline=None)
def test_sens_spec_query_invariants(sample, q):
    # the sensitivity/specificity readouts must hold their invariants on inputs
    # no one chose: sens_at_spec is exactly the reflected at() read, and both
    # readouts keep lower <= estimate <= upper at every query point.
    neg, pos, _seed = sample
    y_true = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    y_score = np.r_[neg, pos]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # small samples / ties may warn
        band = roc_band(y_true, y_score, random_state=0)

    query = np.array([q, 1.0 - q, 0.0, 1.0])
    for got, want in zip(band.sens_at_spec(query), band.at(1.0 - query), strict=True):
        np.testing.assert_array_equal(got, want)

    lo, se, up = band.sens_at_spec(query)
    assert (lo <= se + 1e-12).all()
    assert (se <= up + 1e-12).all()

    lo, sp, up = band.spec_at_sens(query)
    finite = ~np.isnan(sp)
    assert (lo[finite] <= sp[finite] + 1e-12).all()
    assert (sp[finite] <= up[finite] + 1e-12).all()


@pytest.mark.parametrize("n_boot", [100, 250])
def test_envelope_arm_contained_in_boot_range(n_boot):
    # the retained-curve envelope can never escape the pointwise range of
    # the full bootstrap matrix
    rng = np.random.default_rng(3)
    neg = np.sort(rng.normal(0, 1, 40))
    pos = np.sort(rng.normal(1, 1, 40))
    grid = make_grid(40)
    boot = bootstrap_tpr_matrix_numpy(neg, pos, grid_k_indices(grid, 40), n_boot, 4)
    band = assemble_envelope_band(boot, grid, neg, pos, 0.05)
    assert (band.lower_env >= boot.min(axis=0) - 1e-12).all()
    assert (band.upper_env <= boot.max(axis=0) + 1e-12).all()
