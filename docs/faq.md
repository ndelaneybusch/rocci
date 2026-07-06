# FAQ

## Should I pass logits or probabilities?

Either, whichever is easiest. The envelope band is invariant to strictly
monotone transforms of the scores, so logits and their sigmoid give the
*identical* band. Don't calibrate first on rocci's account.

This does not hold for `normal=True`. Binormal fits use the scores
themselves. Typically the logits are the normally distributed piece, and
probabilities will give the wrong thing. This may surprise, since logits and
probabilities yield identical ROC curves — it's only the parametric band
around that ROC that cares about normality. Gotchas of this sort are a big
part of the motivation for rocci's distribution-free default.

## The normality diagnostics stayed quiet — is `normal=True` safe, then?

Quiet is weak evidence, not a certificate. The checks behind
`band.normality` (Shapiro–Francia and D'Agostino K² per class, plus a
probit-linearity check once both classes reach 1000 samples) are calibrated
as a tripwire: they flag roughly 15% of *truly binormal* datasets — a
deliberate cost, because a false alarm is just a warning while a missed
departure is a silent coverage failure. Their detection power grows with
sample size in rough proportion to how fast coverage damage accrues, so
gross departures are caught essentially always. But in *mildly* non-normal
regimes at a few hundred samples per class, a meaningful share of datasets
passes every check while the parametric band still misses the true curve —
there is no diagnostic that can certify binormality from the sample. Use
`normal=True` when you have prior grounds for binormality; when validity
matters and you don't, the distribution-free default needs no such grounds.

## Why is my lower band 0 at small FPR?

Because no distribution-free lower bound exists there. Below FPR ≈
`band.vacuous_below` (roughly $\log(1/\alpha_e)/n_\text{neg}$), the truth is
governed by the extreme tail of the negative distribution, about which your
sample carries almost no information — any method claiming otherwise is
assuming a tail shape. rocci reports the boundary honestly instead. More
negatives shrink the region; see
[Reading the band](guide/reading-the-band.md).

## My scores have lots of ties / are discrete. Is the band still valid?

Yes. Every layer of the envelope is either tie-indifferent or conservative under
ties. Discreteness widens the band but doesn't break it. You'll get a
`TiesWarning` when fewer than half the pooled scores are distinct.

Heavy ties *are* strong evidence against `normal=True`, though: ties have
probability zero under a binormal model.

## Why does `roc_band` refuse my multiclass problem?

Use [`roc_band_ovr`][rocci.roc_band_ovr]: one-vs-rest bands for every class at a
Bonferroni-split level, which gives an exact (conservative) family-wise
guarantee: all m curves are covered jointly at your requested confidence.

And why does `roc_band_ovr` refuse `normal=True`? Because every one-vs-rest
"rest" class is a mixture of the remaining classes, which is structurally the
regime where the binormal band's coverage collapses. We know the parametric
assumption is dangerous here, so we don't provide the option.

## Why is the band wider than the pointwise intervals I'm used to?

Because it makes a stronger statement: the *whole* true (population) curve is
inside with the stated probability. This carries the implication that you can
read off any number of operating points from the rocci band _with no
multiplicity correction_ and each will maintain coverage at the desired alpha.

## Why doesn't `band.auc` match the trapezoid of `band.tpr`?

`band.auc` is the exact Mann–Whitney statistic on the full data (identical to
`sklearn.roc_auc_score`), not an integral of the K-point grid curve — the
grid is an evaluation mesh for the band, not the AUC estimator.

## Is the band reproducible if I allow multi-threading?

`random_state=seed` gives a bit-identical band for the same rocci version,
independent of thread count. Across backends (Rust vs the NumPy fallback)
results agree statistically but not bit-for-bit (they have different RNG
streams).

## Can I use these bands for statistical inference?

Yes. See [Reading the band](guide/reading-the-band.md).

## `confidence=0.8` gave me a band wider than I expected. Why the warning?

Sup-norm simultaneous bands over-cover at low confidence levels. Below 0.90
rocci warns (`LowConfidenceWarning`) that the band is likely to be conservative.
