# Reading the band

## Simultaneous, not pointwise

`band.confidence = 0.95` means: across repeated datasets, the **entire true
ROC curve** lies inside the band 95% of the time. Formally, with $R$ the
population ROC and $[L, U]$ the band,

$$
\Pr\bigl(L(t) \le R(t) \le U(t)\ \text{for all } t \in [0,1]\bigr) \ge 0.95 .
$$

This is strictly stronger than 95% *pointwise* intervals. A pointwise
interval is right about its own FPR 95% of the time — but with hundreds of
grid points, the chance that *every* pointwise interval simultaneously holds
is far below 95%; somewhere along the curve, at least one is essentially
guaranteed to miss. Any claim of the form "the true curve looks like the
drawn one" is implicitly a claim about the whole curve, and only a
simultaneous band supports it.

Practical consequences you may find surprising at first:

- A simultaneous band is **wider** than the pointwise intervals you may be
  used to. That is not looseness; that is the price of the joint statement.
- You may read off **any number of operating points** from the same band —
  `band.at([0.01, 0.05, 0.1])` — with no multiple-comparison correction.
  The correction is already inside.

## The step convention

`band.lower`, `band.tpr`, `band.upper` live on the FPR grid `band.fpr` and
are right-continuous step functions. `band.at(x)` evaluates all three at
arbitrary FPRs with the same convention used to build the band, so what you
query is exactly what was calibrated. Both `plot()` and `at()` share it.

## The vacuous region at tiny FPR

`band.summary()` may say:

> no distribution-free lower bound exists below FPR ≈ 0.0125; increase the
> number of negatives to certify lower FPRs.

This is a real mathematical limit, not a rocci limitation. At FPR below
roughly $1/n_\text{neg}$, whether the true curve is high or low there is
governed by the extreme tail of the negative score distribution — about which
$n_\text{neg}$ samples carry almost no information. Any distribution-free
lower bound must therefore be 0 there. rocci sets `band.lower = 0` in that
region and reports the boundary as `band.vacuous_below` instead of drawing a
confident-looking curve it cannot justify. The upper arm remains informative.

`band.plot(show_vacuous=True)` hatches the region.

## The floors

The bootstrap resamples your data; where the data are nearly degenerate, the
bootstrap can be blind to real uncertainty. Two exact mechanisms repair this:

- the **Wilson rectangle floor** widens the band wherever the bootstrap
  variance collapses below even the binomial noise floor of the empirical
  rates;
- the **Beta order-statistic floor** *lowers* an overconfident lower arm at
  extreme FPR, where the operative uncertainty is horizontal (where the
  threshold's FPR actually sits) and invisible to any variance-based method.

Every grid point of the lower arm records which mechanism produced it in
`band.attribution` (0 = bootstrap envelope, 1 = Beta floor, 2 = Wilson floor,
3 = pinned endpoint), and [Diagnostics](diagnostics.md) shows how to see it.

## The endpoints

`(0, 0)` and `(1, 1)` are on every ROC curve by definition, so the band is
pinned there: `lower[0] = 0`, and at FPR = 1 both arms equal 1 (the true TPR
at FPR 1 is identically 1 — no uncertainty to represent).

## AUC and its interval

- `band.auc` is the exact Mann–Whitney statistic (ties weighted ½) —
  identical to `sklearn.roc_auc_score`.
- `band.auc_ci` is a bootstrap percentile CI for the AUC, recentered so it is
  consistent with `band.auc` even under heavy ties. It is a **pointwise**
  interval for the scalar AUC — separate from, not derived from, the
  simultaneous band.

## Comparing two models

The honest use of bands for model comparison: if model A's *lower* arm sits
above model B's *upper* arm over the FPR range you care about, A dominates B
there at joint confidence. Overlapping bands do **not** prove equivalence —
absence of evidence, as usual.
