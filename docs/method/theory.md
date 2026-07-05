# Theoretical behavior

What can be *guaranteed*, what is *calibrated*, and where the honest edges
are. (Condensed from the method paper's theory. [The envelope
method](envelope.md) describes what is computed; [How rocci is
verified](verification.md) shows how the implementation is held to it.)

## Exact, distribution-free ingredients

Two pieces of the band are exact for every score distribution, at every
sample size:

**The Beta law of extreme thresholds.** For continuous scores, the population
FPR exceedance at the $j$-th largest of $n_0$ negatives is distributed
exactly $\text{Beta}(j,\, n_0 + 1 - j)$ — a rank statement, free of the score
distribution. This is what lets the band make *certified* lower-bound claims
near FPR 0 with a finite-sample, non-asymptotic guarantee, and what fixes the
boundary $q_1$ below which no such claim is possible.

**Wilson binomial floors.** At a fixed threshold, empirical FPR and TPR are
binomial proportions; the Wilson construction gives their exact-level
intervals. Used as the studentization variance floor and, Šidák-corrected, as
the rectangle floor.

Under ties both statements err conservative (the true exceedance is
stochastically smaller than the Beta law), so discreteness degrades the band
toward wider, never toward invalid — verified empirically by a dedicated
discretized-scores calibration cell.

## Rank invariance

The bootstrap kernel and both floors depend on the data only through order
statistics and counts, so the envelope band is invariant under any strictly
monotone transform of the scores: logits, probabilities, ranks — identical
band. Practical consequences: no calibration step is needed before banding,
and the method cannot be gamed or broken by monotone re-scaling. Parametric
bands do not have this property.

## Calibration of the assembled band

The envelope itself (studentize, retain by sup-deviation, take the envelope)
is a bootstrap procedure: its simultaneous coverage is asymptotically correct
and, at finite n, *calibrated* rather than proven — which is why the method
ships with a calibration gate rather than a theorem alone. The gate runs the
assembled band across score distributions chosen to be unkind — binormal,
heavy-tailed (t₃), bimodal negatives, and discretized (heavy-ties) scores —
at n as small as 30 per class, and requires coverage near nominal (from
above, within Monte-Carlo tolerance) with mean width strictly below the
KS/DKW reference band. Coverage *near* nominal is the target: closer is
better, not higher — an over-wide band is a quieter failure than an
under-covering one, but it is still a failure.

Two structural facts explain where the misses can be:

- Coverage violations, when they occur, are small and localized (a single
  grid point escaping briefly), because the sup-studentized retention spends
  the error budget uniformly along the curve rather than at its noisiest
  point.
- Under very heavy discreteness the band is typically conservative (the tie
  direction of both exact ingredients), with residual small under-coverage
  possible at moderate n — the calibration ties cell exists precisely to
  keep this characterized rather than argued.

## Comparison with the fixed-width KS band

The DKW inequality gives a distribution-free simultaneous band with *proven*
finite-sample coverage — rocci uses it as the yardstick, not the product,
because its width is constant: dictated everywhere by the hardest point of
the curve, vacuous at extreme FPR without saying so, and typically far wider
than the envelope across the operating range. The envelope trades the closed
proof for calibrated, locally-adaptive width plus exact repairs where
adaptivity is impossible; the gate enforces that this trade is won (mean
width strictly below KS) while holding coverage.

## The vacuous region is a theorem, not a choice

For any procedure that is valid for **all** continuous score distributions,
the lower band below $t \approx q_1 = \text{Beta}(1, n_0)^{-1}(1 - \alpha_e)$
must be 0: with positive probability under some admissible distribution, the
largest negative sits above every threshold achieving those FPRs, and no
finite sample can exclude it. Increasing $n_\text{neg}$ shrinks $q_1$
hyperbolically ($q_1 \approx \log(1/\alpha_e) / n_0$). rocci reports the
boundary (`band.vacuous_below`, the summary line, the hatched plot region)
instead of drawing an unjustifiable curve.

## Multiclass composition

Bonferroni over $m$ one-vs-rest curves needs no independence: joint coverage
of all curves at family level $1 - \alpha$ follows from per-curve validity at
$1 - \alpha/m$. Because the envelope is distribution-free per curve — in
particular, valid for the *mixture* distributions that one-vs-rest "rest"
classes always are — the guarantee composes exactly. A parametric band loses
precisely this step: rest-class mixtures are the binormal model's
characteristic failure mode, which is why `roc_band_ovr` refuses
`normal=True`.

## Working–Hotelling under misspecification

The WH band's coverage is exact under binormality and degrades
*continuously* with departure from it; because width shrinks as
$O(n^{-1/2})$ while the parametric bias is fixed, coverage **worsens as n
grows** — more data makes the wrong band more confidently wrong (bimodal
negatives drive coverage far below half). This is the structural reason the
diagnostics carry a warning rather than a certificate, and the reason the
distribution-free band is the default.
