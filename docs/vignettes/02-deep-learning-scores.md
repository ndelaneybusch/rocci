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

# Deep-learning scores

Network logits are rarely Gaussian — heavy tails are the norm. This vignette
shows what that does to a parametric ROC band, and demonstrates the
envelope's rank invariance: logits and sigmoid probabilities give the
*identical* band.

The scores are a pre-saved array (so this page's build does not depend on
torch): heavy-tailed logits from two classes, the kind of shape an
overconfident network produces.

```python
import numpy as np
from pathlib import Path

DATA = Path("docs/vignettes/data")   # repo root (the docs build's CWD)
if not DATA.exists():                # running the notebook from docs/vignettes
    DATA = Path("data")

data = np.load(DATA / "dl_scores.npz")
y_true = data["y_true"]
logits = data["logits"].astype(np.float64)
print(f"n = {len(y_true)}, positives = {int(y_true.sum())}")
```

## The envelope band doesn't care about tails

```python
from rocci import roc_band

band = roc_band(y_true, logits, random_state=0)
band.plot()
```

## The parametric band does

Ask for the Working-Hotelling band on the same data, and rocci's normality
diagnostics fire — note the warning:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    wh = roc_band(y_true, logits, normal=True)
print(caught[0].message)
```

The diagnostics behind the warning are attached to the result:

```python
r = wh.normality
print(f"negative class: SF p = {r.neg_sf_pvalue:.2e}, K² p = {r.neg_k2_pvalue:.2e}")
print(f"positive class: SF p = {r.pos_sf_pvalue:.2e}, K² p = {r.pos_k2_pvalue:.2e}")
print(f"negative class: skew = {r.neg_skew:.2f}, excess kurtosis = {r.neg_excess_kurtosis:.2f}")
print(f"probit-probit ROC linearity R² = {r.probit_r2:.4f}")
```

Overlay the two bands. Because this dataset is *simulated* heavy-tailed data
(t₃ classes), we can also draw the true population ROC — which the parametric
band loses:

```python
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

grid = band.fpr
true_roc = 1.0 - student_t.cdf(
    student_t.ppf(1.0 - np.clip(grid, 1e-12, 1), df=3) - 1.6, df=3
)

ax = band.plot(label="95% envelope band")
ax.plot(grid, wh.lower, "--", color="#D55E00", lw=1.2,
        label="Working-Hotelling band")
ax.plot(grid, wh.upper, "--", color="#D55E00", lw=1.2)
ax.plot(grid, true_roc, color="black", lw=1.2, label="true ROC")
ax.legend(loc="lower right", fontsize="small")
```

The failure is worth reading closely, because it is not the one people
usually expect. Misspecification does not merely make a parametric band too
narrow — it biases the band's *center*. The Working-Hotelling band is a
simultaneous band around the best-fitting **binormal** curve, and no binormal
curve has the shape of a t₃ ROC. A t₃ standard deviation (√3 ≈ 1.73) is
mostly tail — the central bulk of the distribution is only about as wide as a
normal with σ ≈ 1.1 — so matching means and standard deviations fits a
binormal model for classes far blurrier, through the mid-range, than these
actually are. The fitted curve lands below the truth exactly where it
matters: the true ROC escapes above the band from FPR ≈ 0.05 to 0.5, and
exits below it at high FPR, while staying inside the envelope at every grid
point. The empirical ROC tracks the truth and escapes the same way. If the
plot looks like the WH band is covering some *other* curve than the one you
computed, that is precisely what is happening: it faithfully covers the
binormal ROC — a curve the data refute.

More data makes this worse, not better: the center bias is a property of the
model, not the sample, so the band narrows as 1/√n around a curve that stays
wrong. And the failure is quiet — on its own, the WH band looks perfectly
plausible. That is why the diagnostics warn rather than certify, and why the
envelope is the default.

## Rank invariance: logits vs probabilities

A sigmoid is strictly monotone, so it cannot change any score's rank — and
the envelope band depends on scores only through ranks:

```python
probs = 1.0 / (1.0 + np.exp(-logits))
band_probs = roc_band(y_true, probs, random_state=0)

assert np.array_equal(band.lower, band_probs.lower)
assert np.array_equal(band.upper, band_probs.upper)
print("envelope band from logits == envelope band from probabilities")
```

No calibration step is needed before banding — pass whichever you have. The
Working-Hotelling band does **not** have this property (the binormal fit
lives in score space), which is one more structural reason to prefer the
default.
