"""Slow statistical calibration gates for the assembled envelope method.

This is the closest the suite comes to validating the headline claim — that the
band achieves *simultaneous* coverage — end to end. Over four data-generating
processes with closed-form population ROCs (binormal, Student-t3, a bimodal
mixture, and a discretized/heavy-tie binormal) at two sample sizes,
it Monte-Carlo estimates the fraction of draws whose band fully contains the true
ROC and checks that estimate against a floor.

Guaranteed. Empirical simultaneous coverage stays at or above 0.906 (the lower
edge of the central 99.9% binomial interval around a 95% target at 250 sims),
and it does so without cheating on width: the mean band is required to be
narrower than the KS/DKW reference band on the same draws, so an all-covering
vacuous band cannot pass. Because the seed sequences are fixed, a regression that
erodes coverage or inflates width changes the numbers deterministically.

Limitations. This is a deterministic regression gate, not a statistical proof:
the coverage estimate is over ``N_SIMS`` draws (default 250, env-tunable) at a
fixed 0.95 target on this specific DGP table, so it can catch a coverage
regression but does not certify coverage on distributions outside the table or
at other confidence levels. The 0.906 floor is intentionally permissive toward
over-coverage — the small-n floor stack may be conservative — so undercoverage
is the failure this guards, and the KS-width check is what rejects the vacuous
escape hatch. Marked ``slow``.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import optimize
from scipy.stats import norm, t

from rocci import roc_band
from rocci._warnings import RocciWarning
from rocci.band.grids import empirical_roc_on_grid, make_grid, step_lookup

CONFIDENCE = 0.95
ALPHA = 1.0 - CONFIDENCE
N_SIMS = int(os.environ.get("ROCCI_CALIBRATION_SIMS", "250"))
N_BOOT = int(os.environ.get("ROCCI_CALIBRATION_BOOT", "2000"))
# The central 99.9% interval of Binomial(250, 0.95)/250 is [0.906, 0.988]. The current
# small-n floor stack can be more conservative than the upper edge; make
# undercoverage a hard failure and let the KS width yardstick reject vacuous
# all-covering bands.
COVERAGE_MIN = 0.906

# Solved offline from AUC(d) = P(pos > neg) = 0.8. The t3 value comes from
# numeric integration over the t density; the mixture value has a closed-form
# AUC but was root-found once to keep the DGP table uniform.
BINORMAL_D = math.sqrt(2.0) * norm.ppf(0.8)
T3_D = 1.5354050089024343
MIXTURE_D = 1.2433497863578125

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Dgp:
    """Data-generating process and its exact population ROC."""

    name: str
    draw: Callable[[np.random.Generator, int], tuple[FloatArray, FloatArray]]
    true_roc: Callable[[FloatArray], FloatArray]


def _clip_unit(p: FloatArray) -> FloatArray:
    return np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)


def _binormal_true_roc(grid: FloatArray) -> FloatArray:
    out = norm.cdf(BINORMAL_D + norm.ppf(np.clip(grid, 1e-15, 1.0 - 1e-15)))
    out[grid <= 0.0] = 0.0
    out[grid >= 1.0] = 1.0
    return _clip_unit(out)


def _t3_true_roc(grid: FloatArray) -> FloatArray:
    out = 1.0 - t.cdf(t.ppf(1.0 - np.clip(grid, 1e-15, 1.0 - 1e-15), df=3) - T3_D, df=3)
    out[grid <= 0.0] = 0.0
    out[grid >= 1.0] = 1.0
    return _clip_unit(out)


def _mixture_cdf(x: FloatArray) -> FloatArray:
    return 0.5 * norm.cdf(x, loc=-1.0, scale=0.5) + 0.5 * norm.cdf(
        x, loc=1.0, scale=0.5
    )


@cache
def _mixture_threshold(prob: float) -> float:
    return optimize.brentq(lambda x: _mixture_cdf(np.asarray(x)) - prob, -8.0, 8.0)


def _mixture_true_roc(grid: FloatArray) -> FloatArray:
    out = np.empty_like(grid, dtype=np.float64)
    interior = (grid > 0.0) & (grid < 1.0)
    out[grid <= 0.0] = 0.0
    out[grid >= 1.0] = 1.0
    if interior.any():
        thresholds = np.array(
            [_mixture_threshold(float(1.0 - p)) for p in grid[interior]],
            dtype=np.float64,
        )
        out[interior] = 1.0 - norm.cdf(thresholds, loc=MIXTURE_D, scale=0.75)
    return _clip_unit(out)


@cache
def _discretized_vertices() -> tuple[FloatArray, FloatArray]:
    h = 0.1
    lo = math.floor((min(0.0, BINORMAL_D) - 5.0) / h)
    hi = math.ceil((max(0.0, BINORMAL_D) + 5.0) / h)
    support = np.arange(hi, lo - 1, -1, dtype=np.float64) * h

    fpr_v = 1.0 - norm.cdf(support - h / 2.0, loc=0.0, scale=1.0)
    tpr_v = 1.0 - norm.cdf(support - h / 2.0, loc=BINORMAL_D, scale=1.0)
    fpr_v = np.concatenate(([0.0], fpr_v, [1.0]))
    tpr_v = np.concatenate(([0.0], tpr_v, [1.0]))
    return fpr_v, tpr_v


def _discretized_true_roc(grid: FloatArray) -> FloatArray:
    fpr_v, tpr_v = _discretized_vertices()
    out = step_lookup(fpr_v, tpr_v, grid)
    out[grid <= 0.0] = 0.0
    out[grid >= 1.0] = 1.0
    return _clip_unit(out)


def _draw_binormal(rng: np.random.Generator, n: int) -> tuple[FloatArray, FloatArray]:
    return rng.normal(0.0, 1.0, n), rng.normal(BINORMAL_D, 1.0, n)


def _draw_t3(rng: np.random.Generator, n: int) -> tuple[FloatArray, FloatArray]:
    return rng.standard_t(3, n), rng.standard_t(3, n) + T3_D


def _draw_bimodal(rng: np.random.Generator, n: int) -> tuple[FloatArray, FloatArray]:
    component = rng.random(n) < 0.5
    neg = np.empty(n, dtype=np.float64)
    neg[component] = rng.normal(-1.0, 0.5, component.sum())
    neg[~component] = rng.normal(1.0, 0.5, (~component).sum())
    return neg, rng.normal(MIXTURE_D, 0.75, n)


def _draw_discretized(
    rng: np.random.Generator, n: int
) -> tuple[FloatArray, FloatArray]:
    neg, pos = _draw_binormal(rng, n)
    return np.round(neg, 1), np.round(pos, 1)


DGPS = (
    Dgp("binormal", _draw_binormal, _binormal_true_roc),
    Dgp("student_t3", _draw_t3, _t3_true_roc),
    Dgp("bimodal_negatives", _draw_bimodal, _mixture_true_roc),
    Dgp("discretized_binormal", _draw_discretized, _discretized_true_roc),
)


def _ks_reference_width(neg: FloatArray, pos: FloatArray, grid: FloatArray) -> float:
    alpha_m = 1.0 - math.sqrt(1.0 - ALPHA)
    c = math.sqrt(math.log(2.0 / alpha_m) / 2.0)
    d0 = c / math.sqrt(len(neg))
    d1 = c / math.sqrt(len(pos))

    upper = np.clip(
        empirical_roc_on_grid(neg, pos, np.clip(grid + d0, 0.0, 1.0)) + d1, 0.0, 1.0
    )
    lower = np.clip(
        empirical_roc_on_grid(neg, pos, np.clip(grid - d0, 0.0, 1.0)) - d1, 0.0, 1.0
    )
    return float(np.mean(upper - lower))


def _seeds(dgp_index: int, n: int, sim: int) -> tuple[np.random.Generator, int]:
    data_seq, boot_seq = np.random.SeedSequence([0xA15, dgp_index, n, sim]).spawn(2)
    boot_seed = int(boot_seq.generate_state(1, dtype=np.uint32)[0])
    return np.random.default_rng(data_seq), boot_seed


@pytest.mark.slow
@pytest.mark.parametrize(
    ("dgp_index", "dgp"), list(enumerate(DGPS)), ids=[dgp.name for dgp in DGPS]
)
@pytest.mark.parametrize("n", [30, 200])
def test_envelope_calibration_gate(dgp_index: int, dgp: Dgp, n: int):
    grid = make_grid(n)
    truth = dgp.true_roc(grid)
    covered = 0
    widths: list[float] = []
    ks_widths: list[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RocciWarning)
        for sim in range(N_SIMS):
            rng, boot_seed = _seeds(dgp_index, n, sim)
            neg, pos = dgp.draw(rng, n)
            y_true = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
            y_score = np.concatenate([neg, pos])

            band = roc_band(y_true, y_score, n_boot=N_BOOT, random_state=boot_seed)
            covered += int(
                np.all((band.lower <= truth + 1e-12) & (truth <= band.upper + 1e-12))
            )
            widths.append(float(np.mean(band.upper - band.lower)))
            ks_widths.append(_ks_reference_width(neg, pos, band.fpr))

    coverage = covered / N_SIMS
    assert coverage >= COVERAGE_MIN
    assert np.mean(widths) < np.mean(ks_widths)
