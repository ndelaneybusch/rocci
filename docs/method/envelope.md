# The envelope method

This page describes what `roc_band(..., normal=False)` — the default —
actually computes. The exact algorithms, including every edge and tie
convention, are the implementation in `python/rocci/band/`, held in place by
the golden-master and oracle test suites (see
[How rocci is verified](verification.md)); this is the readable tour. Throughout, $n_0$ and $n_1$ are the negative/positive class
sizes, $\alpha = 1 - \text{confidence}$, and $R(t)$ is the population TPR at
FPR $t$.

## 1. Grid and empirical curve

The band lives on a uniform FPR grid $t_k = \text{linspace}(0, 1, K)$ with
$K = \min(512,\, n_0 + 1)$ — one point per achievable FPR step, capped. The
empirical ROC $\hat R$ is the standard right-continuous step function with
$\ge$-threshold tie semantics, computed in $O(n \log n)$.

## 2. Bootstrap

For each of $B$ replicates (default 2000), resample both classes with
replacement and evaluate the resampled ROC at every grid point: the TPR at
$t$ is the fraction of resampled positives strictly greater than the
resampled negatives' order statistic at descending index
$k_t = \lfloor t \cdot n_0 \rfloor$. The kernel never sorts a resample — it
tallies draw counts over the pre-sorted originals and reads thresholds and
TPR counts off with linear walks, $O(n_0 + n_1 + K)$ per replicate. Because
each replicate's RNG stream is a pure function of (seed, replicate index),
output is bit-identical regardless of thread count.

The result is a $B \times K$ matrix of bootstrap curves.

## 3. Studentization and retention

A raw pointwise envelope over bootstrap curves would be miscalibrated: the
curve is noisy in some places and pinned in others, so a fixed retention
fraction spends its error budget where variance happens to be large. Instead,
each replicate is scored by its **supremum studentized deviation**

$$
s_b = \max_k \frac{\lvert \hat R_b(t_k) - \hat R(t_k) \rvert}{\hat\sigma(t_k)},
$$

where $\hat\sigma(t_k)$ is the bootstrap standard deviation at $t_k$,
**floored by the Wilson binomial variance** at the empirical rate (so a
grid point where the bootstrap got lucky cannot claim implausibly small
uncertainty), with a small-$\epsilon$ collapse guard for degenerate columns.

The $\lceil (1-\alpha) B \rceil$ replicates with the smallest $s_b$ — the
most typical curves in the studentized metric — are retained, and the band is
their pointwise min/max, clipped to $[0, 1]$. Retention by supremum deviation
is what makes the band simultaneous rather than pointwise: a replicate is
kept or discarded as a *whole curve*.

## 4. Wilson rectangle floor (variance-collapse repair)

The bootstrap can be blind where the data are locally degenerate: with heavy
ties or a flat empirical stretch, resampling reproduces the same TPR almost
every time and the bootstrap variance collapses below even the binomial noise
of estimating a proportion from $n_1$ samples. Wherever the **raw** bootstrap
variance falls below the Wilson variance floor, the band is floored by an
exact 2-D Wilson confidence-rectangle construction at a Šidák-corrected
level (the correction budget adapts to how much of the grid is deficient),
then band monotonicity is enforced. The floor can only widen the band.

## 5. Beta order-statistic floor (extreme-FPR honesty repair)

At extreme FPR the dominant uncertainty is *horizontal*: which FPR the
threshold at a given negative order statistic actually achieves. No
variance-of-TPR mechanism can see this. But for continuous scores an exact,
distribution-free fact is available: the population FPR exceedance at the
$j$-th largest negative is exactly $\text{Beta}(j,\, n_0 + 1 - j)$,
*regardless of the score distribution*. The floor converts this (with a
Bonferroni budget over $2 j_{\max}$ one-sided events, $j_{\max} = 25$) into
certified lower bounds near $t = 0$, applied as a pointwise **minimum**: it
*lowers* an overconfident lower arm to what is actually certifiable — an
honesty repair, never a widening of claims. Under ties the true exceedance is
stochastically smaller than the Beta law, so discrete scores err
conservative.

Below the first jurisdiction boundary $q_1$ no distribution-free lower bound
exists at all; the lower arm is 0 there and the boundary is reported as
`band.vacuous_below`.

## 6. Assembly

The order is load-bearing and fixed:

1. studentized envelope (§3);
2. rectangle floor + band monotonicity (§4);
3. Beta floor (§5) — applied last, no re-monotonization after;
4. pinned endpoints: $L(0) = 0$; $L(1) = U(1) = 1$ (the true ROC at FPR 1 is
   identically 1).

Each grid point of the lower arm records which stage produced it
(`band.attribution`), which is what the
[diagnostics figure](../guide/diagnostics.md) renders.

The assembled pipeline is locked by **golden-master tests**: committed
fixtures recorded from the validated research implementation, which every
build must reproduce within $10^{-6}$. The statistics cannot silently drift.

## 7. AUC

`band.auc` is the exact Mann–Whitney statistic (ties weighted ½ — identical
to `sklearn.roc_auc_score`), and `band.auc_ci` is a bootstrap percentile
interval recentered to be consistent with that estimator even under heavy
ties. Both are reporting-layer quantities, separate from the simultaneous
band.
