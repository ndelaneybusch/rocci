---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Bayesian workflow

Posterior-predictive class probabilities arrive as a *matrix* of draws —
`(draws, n)` from PyMC/arviz, or `(chain, draw, n)`. rocci ingests these
directly with `score_reduce=`. The draws here are pre-saved output of a toy
Bayesian logistic regression (so the docs build depends on neither PyMC nor
torch).

```python
import numpy as np
from pathlib import Path

DATA = Path("docs/vignettes/data")   # repo root (the docs build's CWD)
if not DATA.exists():                # running the notebook from docs/vignettes
    DATA = Path("data")

data = np.load(DATA / "posterior_draws.npz")
y_true = data["y_true"]
draws = data["draws"].astype(np.float64)   # (n_draws, n_samples)
print(f"draws: {draws.shape[0]}, samples: {draws.shape[1]}")
```

## Banding the posterior-mean scores

```python
from rocci import roc_band

band = roc_band(y_true, draws, score_reduce="mean", random_state=0)
band.plot(show_vacuous=True)
```

Passing the raw matrix *without* `score_reduce` is a deliberate error — a
2-D score array is ambiguous, and rocci never guesses:

```python
try:
    roc_band(y_true, draws)
except Exception as err:
    print(err)
```

The reduction is recorded on the result as an INFO note:

```python
print(band.summary())
```

## What this band does — and does not — capture

Be precise about the estimand. Reducing the draws first means the band
quantifies **sampling uncertainty of the ROC of the posterior-mean score** —
"if I drew a fresh dataset of this size and scored it with this (fixed) rule,
where could its true ROC be?" That is exactly the question when the
posterior-mean probability *is* the deployed classifier, which is the common
case.

It does **not** propagate *posterior* uncertainty about the model parameters:
the spread across draws is collapsed by the reduction, not represented in the
band. The two uncertainties answer different questions and coincide only
asymptotically. If you want to see how much the *curve itself* moves across
the posterior, loop the draws:

```python
import matplotlib.pyplot as plt

from rocci.band.grids import empirical_roc_on_grid

ax = band.plot()
rng = np.random.default_rng(0)
for i, d in enumerate(rng.choice(len(draws), size=25, replace=False)):
    neg = np.sort(draws[d][y_true == 0])
    pos = np.sort(draws[d][y_true == 1])
    ax.plot(band.fpr, empirical_roc_on_grid(neg, pos, band.fpr),
            color="#D55E00", alpha=0.15, lw=0.8,
            label="per-draw empirical ROC" if i == 0 else None)
ax.legend(loc="lower right", fontsize="small")
```

The orange cloud (per-draw empirical ROCs on *this* dataset) is posterior
spread; the blue band is sampling uncertainty of the reduced rule. On a
well-behaved problem like this one the cloud sits comfortably inside the
band, but they are conceptually orthogonal — report whichever matches your
claim, and don't present one as the other.
