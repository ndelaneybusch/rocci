# Simulations and validation of the method

The envelope method was validated in a **2.25-million-evaluation simulation
study** — seven data-generating processes, six sample sizes, three confidence
levels — before rocci existed. This page summarizes the study's conclusions.
The full study (figures, theory reports, simulation drivers, and the
reference implementation the golden-master fixtures were recorded from) is in
the [studroc_paper repository](https://github.com/ndelaneybusch/studroc_paper).

## What a band must deliver

A useful simultaneous band needs three properties at once:

1. **Distribution-free** — no parametric commitment about the score
   distributions.
2. **Informative** — width that adapts to where the uncertainty actually is.
3. **Calibrated across families** — coverage near nominal whether the scores
   are Gaussian, heavy-tailed, skewed, or multimodal.

Each classical family fails one of these.

**Working–Hotelling (and its refinements) assumes binormal scores.** The
construction parametrizes the ROC as
$\text{TPR} = \Phi(a + b\,\Phi^{-1}(\text{FPR}))$ and lays a hyperbolic
regression band around it in probit–probit space. When both class score
distributions are Gaussian, this is tight and correct. When they are not,
the band converges to the wrong curve, and the failure scales with sample
size: more data tightens the band around a biased estimate, so coverage
falls exactly when the result looks most authoritative.

**Kolmogorov–Smirnov / DKW is distribution-free but uninformative.** It
places a constant-width strip of half-width
$\varepsilon = \sqrt{\ln(2/\alpha)/(2n)}$ around the curve. It never
under-covers, but the 50% band is nearly as wide as the 95% band, and the
strip is as wide at the steep low-FPR corner as in the flat interior. It is
a safety benchmark, not an answer.

### Non-Gaussian scores are the common case

Binormality is the exception for modern classifiers:

- **Heavy tails.** Deep-network logits and many calibrated probability
  outputs have far heavier tails than a Gaussian; a handful of
  extreme-confidence predictions dominate the low-FPR corner of the ROC,
  precisely where parametric bands are most fragile.
- **Skew and bounded support.** Probability outputs live in $[0, 1]$ and
  pile up near the boundaries; risk scores are often right-skewed. Neither
  is Gaussian on any scale.
- **Multimodality.** When a population mixes subgroups — easy vs. hard
  cases, disease subtypes, distinct fraud patterns — the class score
  distributions are multimodal, producing inflections in the ROC that no
  binormal model can represent.

One non-example: the ROC curve looks identical whether using logits or
probabilities, but the parametric Working-Hotelling band fails on probabilities
(non-normal) but succeeds on binormal logits (normal). It is extremely easy for
a novice practitioner to fall into this trap, because the ROC curve itself is
totally valid when constructed from probabilities. The new default rocci method
is expected to succeed in basically all the cases the ROC curve is
constructable.

## Coverage across the grid

Across the study's coverage heatmaps, the studentized envelope sits at or
slightly above nominal for every distribution at every sample size.
Working–Hotelling holds on the two binormal-compatible families (binormal,
heteroscedastic Gaussian) and fails everywhere else, deepening with $n$: at
n = 10,000 its coverage is 0.02 on Student-t data, 0.13 on logit-normal,
0.23 on bimodal negatives. Off-model, the parametric band does not lose
efficiency; it loses validity.

The failure is not a knife-edge on pathological data — it is a smooth
collapse as the data drift from binormality. As Student-t tails get heavier
or the negative class grows more bimodal, Working–Hotelling coverage slides
continuously from acceptable to near zero, and larger samples make it
worse; the envelope and KS bands hold nominal throughout. No simple
diagnostic defines a safe operating region for the parametric band, which
is why rocci's
[normality diagnostics](../guide/reading-the-band.md#normality-diagnostics)
warn rather than certify.

## Coverage and width, jointly

A band must be judged on both at once: coverage at or above nominal, and,
among bands that clear that bar, the smaller width wins. Plotting per-DGP
coverage against mean band area:

- **KS** is always at ~1.0 coverage but always the widest — safe but
  uninformative, as designed.
- **Working–Hotelling** is competitive only on Gaussian-like data, and even
  there its coverage falls off the bottom of the plot as $n$ grows; on
  heavy-tailed and non-standard data it is near the floor.
- **Wilson rectangles** are tight and well-calibrated at small $n$ but
  drift below nominal as $n$ increases (the Šidák correction mishandles
  many correlated grid points).
- **A pointwise bootstrap** ignores multiplicity and under-covers
  everywhere.

Only the studentized envelope holds high coverage at modest width across
Gaussian-like, heavy-tailed/skewed, and non-standard families, at every
sample size.

## Stability

- **Across AUC.** Coverage holds near or above nominal over the full range
  of true AUC, for every DGP and sample size.
- **Across distribution shape.** Sweeping each DGP's shape parameter — tail
  heaviness, skew strength, variance ratio, mode separation, mixture
  weight — leaves coverage essentially flat above the nominal line.
- **Across sample size.** The envelope holds calibration at all six sample
  sizes on all seven families; Working–Hotelling collapses with $n$ on
  every non-binormal family and the Wilson rectangles drift down
  everywhere.
- **Tighter than KS while calibrated.** The envelope's band area falls from
  90% of the KS band's at n = 10 to 31% at n = 10,000, with calibration
  error flat near the ideal.
- **Small misses.** Residual violations are rare and shallow: the
  conditional miss depth is almost always well under a few points of TPR,
  and the 99th-percentile worst violation stays near zero across all sample
  sizes, where Working–Hotelling and the rectangles climb to 0.7–0.85.

## Ablations

The study removed each component of the hybrid construction
([The envelope method](envelope.md)) in turn; each removal breaks the band
in a specific, predictable way:

- **Without the floors**, the bare bootstrap envelope is not a valid band:
  it fails at both corners — the collapsed-variance plateau and the steep
  low-FPR corner — covering only ~25–35% of the time.
- **The two floors own different corners.** Dropping the Beta floor reopens
  the steep low-FPR corner (worst at high AUC, where that corner matters);
  dropping the Wilson floor reopens the plateau corner, with violation
  rates there climbing to ~0.7.
- **Neither floor alone suffices.** A Beta-only band leaves an unprotected
  gap just beyond the Beta floor's fixed-$j$ jurisdiction; a Wilson-only
  band buys low-FPR coverage only by inflating to vacuous widths there.
- **The bootstrap interior earns its place.** The floors alone give a band
  that is safe at 95% but neither tight nor tunable — it over-covers badly
  at the 50% level and runs wider than the full method. The studentized
  bootstrap provides the adaptive interior width.

## Theory, confirmed

The study's theoretical account ([Theoretical behavior](theory.md) is the
condensed version) made predictions the simulations bore out:

- Asymptotic validity in the interior, with exact finite-sample floors at
  the boundary grid points where the Gaussian approximation does not apply.
- The two corners are governed by different uncertainty models — binomial
  variance at the TPR plateau, order-statistic geometry at the steep
  low-FPR corner — which is why a single mechanism cannot cover both.
- The dominant coverage risk factor is early slope: a steep low-FPR rise
  converts threshold-location error into large vertical misses. The Beta
  floor targets exactly this regime, raising coverage on the previously
  failing high-AUC strata from ~0.77–0.84 to 0.95–0.99.
- Sup-norm simultaneity is insensitive to $\alpha$, so the band over-covers
  at low confidence levels. It is a high-confidence tool: use it near 95%,
  not at 50%.
- The limitation is informativeness, not validity: the lower band is
  vacuous at extreme low FPR because no distribution-free lower bound
  exists there.

## From the study to this package

rocci carries the validation forward rather than assuming it:
[golden-master fixtures](verification.md) recorded from the study's
reference implementation pin the assembled band, and a distilled version of
the coverage-and-width evaluation runs as a required
[calibration gate](verification.md) on every pull request. The figures
behind every claim on this page, and the code to reproduce the full
simulation grid, are in the
[studroc_paper repository](https://github.com/ndelaneybusch/studroc_paper).
