# FAQ

## Should I pass logits or probabilities?

Either — the envelope band is invariant to strictly monotone transforms of
the scores, so logits and their sigmoid give the *identical* band. Don't
calibrate first on rocci's account. (This does not hold for `normal=True`:
the binormal fit lives in score space, so the WH band changes under monotone
transforms — one more reason the envelope is the default.)

## Why is my lower band 0 at small FPR?

Because no distribution-free lower bound exists there. Below FPR ≈
`band.vacuous_below` (roughly $\log(1/\alpha_e)/n_\text{neg}$), the truth is
governed by the extreme tail of the negative distribution, about which your
sample carries almost no information — any method claiming otherwise is
assuming a tail shape. rocci reports the boundary honestly instead. More
negatives shrink the region; see
[Reading the band](guide/reading-the-band.md).

## My scores have lots of ties / are discrete. Is the band still valid?

Yes. Every layer of the envelope is either tie-indifferent or conservative
under ties (the Beta floor's exceedance law errs safe), so discreteness
widens the band rather than breaking it — this is verified empirically by a
dedicated calibration cell, not just argued. You'll get a `TiesWarning`
stating exactly this when fewer than half the pooled scores are distinct.
Heavy ties *are* strong evidence against `normal=True`, though: ties have
probability zero under a binormal model.

## Why does `roc_band` refuse my multiclass problem?

There is no single ROC curve for m > 2 classes — any single curve would hide
an arbitrary choice of reduction. Use
[`roc_band_ovr`][rocci.roc_band_ovr]: one-vs-rest bands for every class at a
Bonferroni-split level, which gives an exact (conservative) family-wise
guarantee — all m curves covered jointly at your requested confidence, no
independence assumptions. And why does `roc_band_ovr` refuse `normal=True`?
Because every one-vs-rest "rest" class is a mixture of the remaining classes,
which is structurally the regime where the binormal band's coverage
collapses; rocci won't automate its proven worst case.

## Why is the band wider than the pointwise intervals I'm used to?

Because it makes a stronger statement: the *whole* curve is inside with the
stated probability, so you can read off any number of operating points with
no multiplicity correction. Pointwise intervals are individually right and
jointly almost surely wrong somewhere along the curve.

## Why doesn't `band.auc` match the trapezoid of `band.tpr`?

`band.auc` is the exact Mann–Whitney statistic on the full data (identical to
`sklearn.roc_auc_score`), not an integral of the K-point grid curve — the
grid is an evaluation mesh for the band, not the AUC estimator.

## Is the band reproducible?

`random_state=seed` gives a bit-identical band for the same backend and rocci
version, independent of thread count. Across backends (Rust vs the NumPy
fallback) results agree statistically but not bit-for-bit — different RNG
streams by design.

## Can I compare two classifiers with these bands?

If A's lower arm sits above B's upper arm over the FPR range you care about,
A dominates there at joint confidence. Overlap does not prove equivalence.
For a sharper paired comparison a dedicated two-sample procedure would be
needed; rocci does not currently ship one.

## `confidence=0.8` gave me a band wider than I expected. Why the warning?

Sup-norm simultaneous bands intentionally over-cover at low confidence
levels — the envelope's calibration target is the high-confidence regime.
Below 0.90 rocci warns (`LowConfidenceWarning`) that the nominal level
understates the width you'll get.
