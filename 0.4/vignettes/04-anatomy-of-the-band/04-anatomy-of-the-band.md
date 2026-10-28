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

# Anatomy of the band

Every rocci band can explain itself: each point of the lower arm records
which mechanism produced it, and `plot_diagnostics()` renders the "why did my
band do that here" view. This vignette walks the anatomy across sample sizes
at high AUC — the regime where the interesting region (extreme FPR) actually
matters.

```python
import numpy as np
from scipy.stats import norm

from rocci import roc_band

def high_auc_data(n, auc=0.95, seed=0):
    d = np.sqrt(2.0) * norm.ppf(auc)
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)]
    s = np.r_[rng.normal(0, 1, n), rng.normal(d, 1, n)]
    return y, s
```

## The diagnostics figure

```python
y, s = high_auc_data(n=500)
band = roc_band(y, s, random_state=0)
band.plot_diagnostics()
```

**Left panel** — the band, with the lower arm color-coded:

- *uncolored* stretches: the plain bootstrap envelope carried the bound;
- *yellow*: the **Beta order-statistic floor** — at extreme FPR the dominant
  uncertainty is horizontal (where a threshold's FPR actually sits), which no
  variance-based method can see; the floor replaces an overconfident bound
  with the exactly-certified one;
- *green*: the **Wilson rectangle floor** — fires where the bootstrap
  variance collapsed below even binomial noise (flat stretches, near-ties);
- the hatched region below `vacuous_below`: no distribution-free lower bound
  exists there at all.

**Right panel** — the variance channels that drive the floor gate: raw
bootstrap variance vs the Wilson floor, log scale, with each floor's active
jurisdiction shaded.

## The same anatomy across n

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
for ax, n in zip(axes, (50, 500, 5000)):
    y, s = high_auc_data(n)
    b = roc_band(y, s, random_state=0)
    b.plot(ax=ax, show_vacuous=True)
    ax.set_title(f"n = {n} per class")
```

Read left to right:

- **n = 50**: the band is wide, the vacuous region is a visible slice of the
  FPR axis, and the floors own much of the lower arm — honesty dominating.
- **n = 500**: the envelope takes over most of the curve; the vacuous region
  shrinks toward the origin.
- **n = 5000**: a tight band; floors persist only in the extreme corner,
  because the $\text{Beta}(1, n_0)$ boundary shrinks like $1/n_0$ but never
  reaches zero.

## Attribution as data

The color coding is just `band.attribution` (0 = bootstrap envelope, 1 =
Beta floor, 2 = Wilson floor, 3 = pinned endpoint):

```python
for n in (50, 500, 5000):
    y, s = high_auc_data(n)
    b = roc_band(y, s, random_state=0)
    codes, counts = np.unique(b.attribution, return_counts=True)
    share = {int(c): f"{100 * k / len(b.attribution):.0f}%"
             for c, k in zip(codes, counts)}
    print(f"n={n:>5}: attribution shares {share}, "
          f"vacuous below FPR={b.vacuous_below:.4f}")
```

The summary prints the same story in words — `floor jurisdictions:` counts
the points each floor owns:

```python
print(roc_band(*high_auc_data(50), random_state=0).summary())
```
