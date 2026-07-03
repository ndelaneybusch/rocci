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


def binormal_dataset(
    n_neg: int,
    n_pos: int,
    auc: float = 0.8,
    seed: int = 0,
    tie_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_score)`` for the public API, from ``binormal_scores``.

    Negatives are labeled 0, positives 1; the two are concatenated into the
    flat ``(y_true, y_score)`` pair that ``roc_band`` ingests.
    """
    neg, pos = binormal_scores(n_neg, n_pos, auc=auc, seed=seed, tie_step=tie_step)
    y_true = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    y_score = np.concatenate([neg, pos])
    return y_true, y_score
