# Simulations and validation of the method

The envelope method was validated in a **2.25-million-evaluation simulation
study** spanning seven data-generating processes, six sample sizes, and three
confidence levels, before rocci existed. This page marches through the
study's conclusions — the case that the default band is the right default.
The full study — figures, theory reports, simulation drivers, and the
reference implementation the golden-master fixtures were recorded from —
lives in the
[studroc_paper repository](https://github.com/ndelaneybusch/studroc_paper).

## The problem with the methods we have

If you want a band around an ROC curve today, you essentially choose between
two families, and both have a fatal flaw.

**Working–Hotelling (and its refinements) assume the scores are binormal.**
The construction parametrizes the ROC as
$\text{TPR} = \Phi(a + b\,\Phi^{-1}(\text{FPR}))$ and lays a hyperbolic
regression band around it in probit–probit space. When both class score
distributions really are Gaussian, this is excellent: tight bands, correct
coverage. When they are not, the band converges confidently to *the wrong
curve*. Worse, the failure scales with sample size — more data tightens the
band around a biased estimate, so coverage drops toward zero exactly when you
trust the result most.

**The Kolmogorov–Smirnov / DKW fixed-width band is distribution-free but
uninformative.** It places a constant-width strip of half-width
$\varepsilon = \sqrt{\ln(2/\alpha)/(2n)}$ around the curve. It never fails to
cover — but it covers *too much*. The 50% band is nearly as wide as the 95%
band, the strip has the same width at the steep low-FPR corner as in the flat
interior, and it tells you almost nothing about *where* the true curve
actually lies. It is a safety benchmark, not an answer.

What is actually needed is a band that is (i) **distribution-free**, like KS,
making no parametric commitment about the score distributions; (ii)
**substantially more informative than KS**, with width that adapts to where
the uncertainty really is; and (iii) **well calibrated across diverse
distribution families**, holding coverage near nominal whether the scores are
Gaussian, heavy-tailed, skewed, or multimodal.

### Why "alternative distributions" is the realistic case, not the exotic one

Binormality is the exception, not the rule, for modern classifiers:

- **Heavy tails.** Deep network logits and many calibrated probability
  outputs have far heavier tails than a Gaussian; a handful of
  extreme-confidence predictions dominate the low-FPR corner of the ROC,
  which is precisely where parametric bands are most fragile.
- **Skew and bounded support.** Probability outputs live in $[0, 1]$ and pile
  up near the boundaries; risk scores are often right-skewed. Neither is
  Gaussian on any scale.
- **Multimodality.** When a population mixes subgroups — easy vs. hard cases,
  multiple disease subtypes, distinct fraud patterns — the negative or
  positive score distribution is multimodal, producing genuine inflections in
  the ROC that no binormal model can represent.

A clarifying non-example: choosing to build the band on **logit** scores
rather than **probability** scores is *not* a reason to worry about
distributional assumptions. The ROC curve is invariant to any monotone
rescaling of the score, so the probability-vs-logit choice does not move the
curve at all and cannot, by itself, motivate a distribution-free method. The
motivation is the genuine shape of the underlying class-conditional
distributions — heavy tails, skew, multimodality — not the coordinate system
you happen to plot them in.

## How badly the classical bands fail

The contrast is stark. Across the study's coverage heatmaps, the studentized
envelope is at or slightly above nominal for **every** distribution at
**every** sample size. Working–Hotelling is fine on the two
binormal-compatible families (binormal, heteroscedastic Gaussian) but fails
everywhere else, and the failure *deepens with n*: on Student-t data its
coverage falls to 0.02 at n = 10,000, on logit-normal to 0.13, on bimodal
negatives to 0.23. The parametric band does not merely lose efficiency
off-model — it loses validity, catastrophically and irreversibly.

This is not a knife-edge that only trips on pathological data. It is a
smooth, continuous collapse as the data drift away from binormality: as
Student-t tails get heavier or the negative class becomes more clearly
bimodal, Working–Hotelling coverage slides continuously from acceptable to
near zero, and larger sample sizes make it *worse*. The envelope and KS bands
ride flat along the top throughout. There is no safe operating region for the
parametric band defined by a simple diagnostic — any departure from
binormality is paid for in coverage. (This is why rocci's
[normality diagnostics](../guide/diagnostics.md) warn rather than certify.)

## Coverage *and* tightness, on every family

The right way to judge a band is jointly: coverage must be at or above
nominal, and among the bands that clear that bar, smaller width is better.
Plotting per-DGP coverage against mean band area, the competitors each fail a
requirement:

- **KS** is always at ~1.0 coverage but always the widest — safe but
  uninformative, exactly as designed.
- **Working–Hotelling** is competitive only on Gaussian-like data, and even
  there its coverage falls off the bottom of the plot as n grows; on
  heavy-tailed and non-standard data it is near the floor.
- **Wilson rectangles** are tight and well-calibrated at small n but drift
  below nominal as n increases (the Šidák correction mishandles many
  correlated grid points).
- **A pointwise bootstrap** ignores multiplicity entirely and under-covers
  everywhere.

Only the studentized envelope occupies the desirable corner — high coverage,
modest width — across Gaussian-like, heavy-tailed/skewed, and
non-standard-shape families alike, at every sample size.

## The properties that make it trustworthy

- **Stable across AUC.** Coverage holds near or above nominal across the full
  range of true AUC, for every DGP and sample size — it does not degrade as
  the curve gets steeper.
- **Stable across distribution shape.** Sweeping each DGP's shape parameter —
  tail heaviness, skew strength, variance ratio, mode separation, mixture
  weight — leaves coverage essentially flat above the nominal line.
- **Stable across sample size, where competitors are not.** The envelope
  holds calibration across all six sample sizes on all seven families, while
  Working–Hotelling collapses with n on every non-binormal family and the
  Wilson rectangles drift down everywhere.
- **Far tighter than KS, while staying calibrated.** The envelope's band area
  falls from 90% of the KS band's at n = 10 to 31% at n = 10,000 — it buys
  most of KS's safety at roughly a third of the width — while its calibration
  error stays flat near the ideal.
- **When it misses, it misses small.** No finite-sample method is perfect,
  but the envelope's residual violations are tiny: the conditional miss depth
  is almost always well under a few points of TPR, and the 99th-percentile
  worst violation stays near zero across all sample sizes, where
  Working–Hotelling and the rectangles climb to 0.7–0.85.

## Ablations: every component is load-bearing

The hybrid construction ([The envelope method](envelope.md)) is not
over-engineered — the study removed each piece in turn, and each removal
breaks the band in a specific, predictable way:

- **Without the floors, the bare bootstrap envelope is not a valid band at
  all.** It fails at both corners — the collapsed-variance plateau and the
  steep low-FPR corner — covering only ~25–35% of the time.
- **The two floors own different corners.** Dropping the Beta floor reopens
  the steep low-FPR corner (and the damage concentrates at high AUC, where
  that corner matters); dropping the Wilson floor reopens the plateau corner,
  with violation rates there climbing to ~0.7.
- **Neither floor alone suffices.** A Beta-only band leaves an unprotected
  gap just beyond the Beta floor's fixed-$j$ jurisdiction; a Wilson-only band
  buys low-FPR coverage only by inflating the band to vacuous widths there.
  The hybrid is the one configuration that is neither leaky nor vacuous.
- **The bootstrap interior earns its place too.** Replacing it with the
  floors alone yields a band that is safe at 95% but neither tight nor
  tunable — it over-covers badly at the 50% level and runs wider than the
  full method. The studentized bootstrap is what gives the interior its
  adaptive, informative width.

## What the theory predicted — and simulation confirmed

The study's theoretical account ([Theoretical behavior](theory.md) is the
condensed version) made predictions the simulations then bore out:

- Asymptotic validity in the interior, with exact finite-sample floors at
  the boundary grid points where the Gaussian approximation does not apply.
- The two corners are governed by two different, complementary uncertainty
  models — binomial variance at the TPR plateau, order-statistic geometry at
  the steep low-FPR corner — which is why a single mechanism cannot work.
- The real coverage risk factor is early slope: a steep low-FPR rise converts
  threshold-location error into large vertical misses. The Beta floor targets
  exactly this regime, raising coverage on the previously failing high-AUC
  strata from ~0.77–0.84 to 0.95–0.99.
- It is a high-confidence tool: sup-norm simultaneity is inherently
  insensitive to $\alpha$, so the band over-covers at low confidence levels.
  Use it near 95%, not at 50%.
- The honest limitation is informativeness, not validity: the lower band is
  vacuous at extreme low FPR because no distribution-free lower bound exists
  there — the band declines to certify what cannot be certified.

## From the study to this package

rocci carries the validation forward rather than assuming it:
[golden-master fixtures](verification.md) recorded from the study's reference
implementation pin the assembled band bit-for-bit, and a distilled version of
the coverage-and-width evaluation runs as a required
[calibration gate](verification.md) on every pull request. The figures behind
every claim on this page, and the code to reproduce the full simulation grid,
are in the
[studroc_paper repository](https://github.com/ndelaneybusch/studroc_paper).
