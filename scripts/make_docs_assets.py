"""Regenerate the committed docs data files and the README/home hero figure.

Deterministic (fixed seeds). The vignettes deliberately consume *pre-saved*
arrays (docs/vignettes/data/) so the docs build depends on neither torch nor
PyMC; this script is their provenance.

Usage: uv run python scripts/make_docs_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "vignettes" / "data"
ASSETS = ROOT / "docs" / "assets"


def deep_learning_scores() -> None:
    """Heavy-tailed 'network logits': t3 classes, the WH failure regime."""
    rng = np.random.default_rng(20260703)
    n = 400
    neg = student_t.rvs(df=3, size=n, random_state=rng)
    pos = student_t.rvs(df=3, size=n, random_state=rng) + 1.6
    y_true = np.r_[np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)]
    logits = np.r_[neg, pos].astype(np.float32)
    np.savez_compressed(DATA / "dl_scores.npz", y_true=y_true, logits=logits)


def posterior_draws() -> None:
    """Posterior-predictive probabilities from a toy Bayesian logistic model."""
    rng = np.random.default_rng(20260704)
    n, n_draws = 250, 160
    x = rng.normal(0.0, 1.0, n)
    p_true = 1.0 / (1.0 + np.exp(-(0.3 + 1.4 * x)))
    y_true = (rng.random(n) < p_true).astype(np.int8)
    # posterior over (a, b) around the truth, as an MCMC sampler would give
    a = rng.normal(0.3, 0.25, n_draws)
    b = rng.normal(1.4, 0.30, n_draws)
    draws = 1.0 / (1.0 + np.exp(-(a[:, None] + b[:, None] * x[None, :])))
    np.savez_compressed(
        DATA / "posterior_draws.npz", y_true=y_true, draws=draws.astype(np.float32)
    )


def hero_figure() -> None:
    """Envelope vs Working-Hotelling on heavy-tailed scores, with the truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rocci import roc_band

    with np.load(DATA / "dl_scores.npz") as z:
        y_true, logits = z["y_true"], z["logits"].astype(np.float64)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        envelope = roc_band(y_true, logits, random_state=0)
        wh = roc_band(y_true, logits, normal=True)

    grid = envelope.fpr
    true_roc = 1.0 - student_t.cdf(
        student_t.ppf(1.0 - np.clip(grid, 1e-12, 1), df=3) - 1.6, df=3
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    envelope.plot(ax=ax, label="95% envelope band (rocci default)")
    ax.plot(
        grid,
        wh.lower,
        color="#D55E00",
        linewidth=1.2,
        linestyle="--",
        label="Working-Hotelling band (misses the truth)",
    )
    ax.plot(grid, wh.upper, color="#D55E00", linewidth=1.2, linestyle="--")
    ax.plot(grid, true_roc, color="black", linewidth=1.2, label="true ROC")
    ax.legend(loc="lower right", fontsize="small")
    ax.set_title("Heavy-tailed scores: distribution-free vs binormal")
    fig.savefig(ASSETS / "hero.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    deep_learning_scores()
    posterior_draws()
    hero_figure()
    for f in sorted([*DATA.iterdir(), ASSETS / "hero.png"]):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1024:.0f} KB")
