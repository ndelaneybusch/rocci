# Diagnostics

Every band can explain itself. `band.plot_diagnostics()` renders the "why did
my band do that here" figure; `roc_band(..., diagnostics=True)` renders it
immediately at construction (notebook convenience). The underlying data —
`band.attribution`, and the variance channels — are always stored on the
result whether or not you plot.

## Envelope path: floor attribution

Two panels:

1. **The band, with the lower arm color-coded by attribution.** Uncolored
   stretches are the plain bootstrap envelope. Yellow marks the **Beta
   order-statistic floor** (extreme-FPR honesty repair), green the **Wilson
   rectangle floor** (variance-collapse repair). Jurisdiction boundaries are
   marked, and the vacuous region is hatched.
2. **Variance channels vs FPR** (log scale): the raw bootstrap variance
   against the Wilson variance floor. Wherever the raw variance dips below
   the binomial floor, the bootstrap is under-representing real uncertainty
   and the rectangle floor fires — the shaded intervals show each floor's
   active region.

Typical readings:

- **Yellow at the far left** is normal, and more of it at small
  $n_\text{neg}$: the Beta floor owns the extreme-FPR region where the
  bootstrap cannot see horizontal threshold uncertainty.
- **Green patches** appear where the empirical curve is locally flat or the
  data nearly degenerate (heavy ties, tiny classes) — the bootstrap variance
  collapsed and was floored.
- **Mostly uncolored** means the bootstrap envelope alone carried the band —
  the large-sample, well-behaved regime.

The [anatomy vignette](../vignettes/04-anatomy-of-the-band.md) walks the
attribution across n ∈ {50, 500, 5000} at high AUC.

## Working-Hotelling path: normality evidence

For `normal=True` bands the diagnostics figure shows the band plus the
normality evidence: a normal QQ plot per class and the probit–probit ROC
linearity fit. Under binormality, $\Phi^{-1}(\text{TPR})$ vs
$\Phi^{-1}(\text{FPR})$ is a straight line; curvature is exactly the model
failing.

The same evidence is available programmatically on every WH band:

```python
band = roc_band(y, s, normal=True)
band.normality.neg_pvalue      # per-class normality test p-values
band.normality.pos_pvalue
band.normality.probit_r2       # OLS R² of the probit-probit ROC interior
band.normality.suspect         # True => a NormalityWarning was emitted
band.normality.warning         # the exact text
```

`suspect` fires when either class p-value drops below 0.10 or the probit R²
falls below 0.98. Two things to keep in mind:

- The thresholds are deliberately jumpy: WH coverage degrades *continuously*
  with departures from binormality and worsens with n, so there is no safe
  region the diagnostics could certify. They can only fail to reject.
- Heavy ties are flagged in the warning text as structurally incompatible
  with the binormal model (ties have probability zero under it).

A worked "bad WH" example — heavy-tailed logits where the diagnostics fire
and the parametric band visibly loses the true curve — is the
[deep-learning vignette](../vignettes/02-deep-learning-scores.md).
