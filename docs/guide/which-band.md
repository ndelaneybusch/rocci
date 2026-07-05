# Which band should I use?

**Short answer: the default.** `roc_band(y_true, y_score)` gives the
distribution-free envelope band. Reach for `normal=True` only when you have
good reason to believe both score distributions are Gaussian and you want the
tighter parametric band — and even then, rocci will check the assumption and
warn you when it looks doubtful.

## The three families, and what goes wrong

**Parametric (Working–Hotelling).** Assume each class's scores are normal,
fit means and variances, and propagate the fit uncertainty into a band. When
binormality holds this is the tightest honest band available. When it fails,
it fails *quietly and increasingly*: coverage degrades continuously with the
departure from normality and **gets worse as n grows**, because the band
narrows around a systematically wrong curve. With bimodal negatives, coverage
collapses far below nominal. There is no safe diagnostic region — a passed
normality test at small n mostly reflects the test's low power. The
[hero figure](../index.md) shows the failure on heavy-tailed scores.

**Fixed-width nonparametric (KS/DKW).** Apply a Kolmogorov–Smirnov-style
margin to both empirical CDFs. Assumption-free and simple, but the width is
constant everywhere, dictated by the hardest part of the curve — so the band
is far wider than necessary across most of the ROC, and at extreme FPR it is
*vacuous* (spanning `[0, 1]`) without saying so. rocci's calibration gate
uses the KS band as the width yardstick the envelope must beat.

**Studentized bootstrap envelope (rocci's default).** Bootstrap the ROC,
score each replicate curve by its worst studentized deviation, keep the most
typical `(1 − α)` fraction, and take their pointwise envelope. Studentization
lets the band be narrow where the curve is stable and wide where it is not.
Two exact small-sample repairs handle the places the bootstrap is blind
(see [The envelope method](../method/envelope.md)), and the result is
calibrated across score distributions with **no assumptions on the score
distribution at all**.

The numbers behind all three verdicts — coverage heatmaps, joint
coverage-vs-width comparisons, and component ablations from a
2.25-million-evaluation simulation study — are in
[Simulations and validation](../method/simulations.md).

## Properties worth knowing

**Rank invariance.** The envelope band depends on the scores only through
their ranks. Logits, sigmoid probabilities, or any strictly monotone
transform of the same scores give the *identical* band. The WH band changes
under the same transforms — the parametric fit lives in score space. The
[deep-learning vignette](../vignettes/02-deep-learning-scores.md) demonstrates
both facts on the same data.

**Multiclass composition.** For one-vs-rest multiclass bands
(`roc_band_ovr`), the envelope *composes*: it is distribution-free per curve,
so a Bonferroni split of α across the m classes yields an exact (conservative)
family-wise guarantee with no independence assumption. WH *anti-composes*:
each one-vs-rest "rest" class is a mixture of the remaining classes — which
is structurally the bimodal-negatives regime where its coverage collapses.
For this reason `roc_band_ovr(normal=True)` refuses to run rather than
automate the parametric band's proven worst case.

**Ties are safe.** Every layer of the envelope is either tie-indifferent or
conservative under ties, so discrete scores widen the band slightly rather
than break it. Ties have probability zero under a binormal model, so heavy
ties are also *evidence against* `normal=True` — the diagnostics say so
explicitly.

## When `normal=True` is reasonable

- Scores are well-modeled as Gaussian per class (e.g. calibrated continuous
  biomarkers), and
- the [normality diagnostics](diagnostics.md) come back clean, and
- you accept that the diagnostics can only fail to reject — they cannot
  certify.

You get a visibly tighter band. rocci attaches the full
`NormalityReport` to the result either way, and emits a `NormalityWarning`
when either class fails a normality test or the ROC's probit–probit plot
bends.
