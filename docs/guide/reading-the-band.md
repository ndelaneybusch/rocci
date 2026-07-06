# Reading the band

## Simultaneous, not pointwise

`band.confidence = 0.95` means: across repeated datasets, the **entire true
ROC curve** lies inside the band 95% of the time. With $R$ the population ROC
and $[L, U]$ the band,

$$
\Pr\bigl(L(t) \le R(t) \le U(t)\ \text{for all } t \in [0,1]\bigr) \ge 0.95 .
$$

This is strictly stronger than 95% *pointwise* intervals. Each pointwise
interval is right about its own FPR 95% of the time, but with hundreds of
grid points, at least one is almost guaranteed to miss somewhere along the
curve. Any claim of the form "the true curve looks like the drawn one" is a
claim about the whole curve, and only a simultaneous band supports it.

Two consequences:

- A simultaneous band is **wider** than pointwise intervals. That is the
  price of the joint statement, not looseness.
- You may read off **any number of operating points** from the same band —
  `band.at([0.01, 0.05, 0.1])` — with no multiple-comparison correction.
  The correction is already inside.

## The step convention

`band.lower`, `band.tpr`, and `band.upper` live on the FPR grid `band.fpr`
as right-continuous step functions. `band.at(x)` evaluates all three at
arbitrary FPRs with the same convention used to build the band, so what you
query is exactly what was calibrated. `plot()` shares the convention.

## The vacuous region at tiny FPR

`band.summary()` may say:

> no distribution-free lower bound exists below FPR ≈ 0.0125; increase the
> number of negatives to certify lower FPRs.

This is a mathematical limit, not a rocci limitation. At FPR below roughly
$1/n_\text{neg}$, the true curve is governed by the extreme tail of the
negative score distribution — about which $n_\text{neg}$ samples carry almost
no information. Any distribution-free lower bound must therefore be 0 there.
rocci sets `band.lower = 0` in that region and reports the boundary as
`band.vacuous_below`. The upper arm remains informative.
`band.plot(show_vacuous=True)` hatches the region.

## The floors

The bootstrap resamples your data; where the data are nearly degenerate, it
can be blind to real uncertainty. Two exact mechanisms repair this:

- the **Wilson rectangle floor** widens the band wherever the bootstrap
  variance collapses below the binomial noise floor of the empirical rates;
- the **Beta order-statistic floor** *lowers* an overconfident lower arm at
  extreme FPR, where the operative uncertainty is horizontal (where the
  threshold's FPR actually sits) and invisible to any variance-based method.

Each grid point of the lower arm records which mechanism produced it in
`band.attribution` (0 = bootstrap envelope, 1 = Beta floor, 2 = Wilson floor,
3 = pinned endpoint); [Floor attribution](#floor-attribution) shows how to
visualize it.

## The endpoints

`(0, 0)` and `(1, 1)` are on every ROC curve by definition, so the band is
pinned there: `lower[0] = 0`, and at FPR = 1 both arms equal 1 (the true TPR
at FPR 1 is identically 1 — no uncertainty to represent).

## AUC and its interval

- `band.auc` is the exact Mann–Whitney statistic (ties weighted ½) —
  identical to `sklearn.roc_auc_score`.
- `band.auc_ci` is a bootstrap percentile CI for the AUC, recentered so it
  stays consistent with `band.auc` under heavy ties. It is a **pointwise**
  interval for the scalar AUC — separate from, not derived from, the
  simultaneous band.

## Confidence bands as statistical inference

If any part of the diagonal identity line (i.e. a perfectly uninformative
classifier) lies outside of the confidence band, you may reject the null
hypothesis (of an uninformative ROC curve) at the chosen alpha.

## Comparing two models

If model A's *lower* arm sits above model B's *upper* arm over the FPR range
you care about, A dominates B there at joint confidence. Overlapping bands do
**not** establish equivalence.

You may also compare an empirical ROC to a reference ROC. If any part of the
reference ROC lies outside of the empirical confidence band, you may reject the
null hypothesis that they are equivalent at the chosen alpha.

## Diagnostics

Every band can explain itself. `band.plot_diagnostics()` renders the diagnostics
figure; `roc_band(..., diagnostics=True)` renders it at construction. The
underlying data (`band.attribution`) is stored on the result whether or not you
plot.

### Floor attribution

For envelope bands, the figure has two panels:

1. **The band, with the lower arm color-coded by attribution.** Uncolored
   stretches are the plain bootstrap envelope; yellow marks the Beta
   order-statistic floor, green the Wilson rectangle floor. Jurisdiction
   boundaries are marked and the vacuous region is hatched.
2. **Variance channels vs FPR** (log scale): the raw bootstrap variance
   against the Wilson variance floor. Wherever the raw variance dips below
   the binomial floor, the bootstrap is under-representing real uncertainty
   and the rectangle floor fires; shading shows each floor's active region.

Typical readings:

- **Yellow at the far left** is normal, and grows at small $n_\text{neg}$:
  the Beta floor owns the extreme-FPR region where the bootstrap cannot see
  horizontal threshold uncertainty.
- **Green patches** appear where the empirical curve is locally flat or the
  data nearly degenerate (heavy ties, tiny classes) — the bootstrap variance
  collapsed and was floored.
- **Mostly uncolored** means the bootstrap envelope alone carried the band —
  the large-sample, well-behaved regime.

The [anatomy vignette](../vignettes/04-anatomy-of-the-band.md) walks the
attribution across n ∈ {50, 500, 5000} at high AUC.

### Normality diagnostics

For `normal=True` bands the figure shows the band plus the normality
evidence: a normal QQ plot per class and the probit–probit ROC linearity
fit. Under binormality, $\Phi^{-1}(\text{TPR})$ vs $\Phi^{-1}(\text{FPR})$
is a straight line; curvature is exactly the model failing.

The same evidence is available programmatically on every Working–Hotelling
band:

```python
band = roc_band(y, s, normal=True)
band.normality.neg_pvalue      # smallest per-class check p-value
band.normality.pos_pvalue      # (Shapiro-Francia and D'Agostino K² each run
                               #  where valid; *_sf_* / *_k2_* hold the details,
                               #  *_skew / *_excess_kurtosis the effect sizes)
band.normality.probit_r2       # OLS R² of the probit-probit ROC interior
band.normality.suspect         # True => a NormalityWarning was emitted
band.normality.warning         # the exact text
```

`suspect` fires when **any** check trips — a class-check p-value below 0.02,
or the probit R² below 0.98 once both classes reach 1000 samples (below that
the fixed R² threshold is noise on truly binormal data, so it is reported but
never triggers). The thresholds sit at the balance point that maximizes the
gate's agreement (Matthews correlation) with actual Working–Hotelling
coverage failures across sample sizes and departure families, flagging
roughly 4–8% of *truly binormal* datasets along the way. Even so, no
operating point certifies binormality: coverage degrades *continuously* with
departures and worsens with n, so the diagnostics can only fail to reject. A quiet gate is
weak evidence: in mildly heavy-tailed regimes a meaningful share of datasets
pass every check while the parametric band misses the true curve. Heavy ties are flagged in the warning text as
structurally incompatible with the binormal model (ties have probability zero
under it).

For a worked failure — heavy-tailed logits where the diagnostics fire and the
parametric band visibly loses the true curve — see the
[deep-learning vignette](../vignettes/02-deep-learning-scores.md).
