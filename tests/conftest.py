"""Shared test data helpers: realistic score distributions, not toys."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def binormal_scores(
    n_neg: int,
    n_pos: int,
    auc: float = 0.8,
    seed: int = 0,
    tie_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw binormal class scores with a target population AUC.

    ``tie_step`` rounds scores to that granularity, producing the heavy-ties
    regime (the paper's discretized-binormal cell).
    """
    d = np.sqrt(2.0) * norm.ppf(auc)
    rng = np.random.default_rng(seed)
    neg = rng.normal(0.0, 1.0, n_neg)
    pos = rng.normal(d, 1.0, n_pos)
    if tie_step is not None:
        neg = np.round(neg / tie_step) * tie_step
        pos = np.round(pos / tie_step) * tie_step
    return neg, pos
