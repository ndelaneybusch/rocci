# rocci — Reference Appendix (Normative Algorithms)

Companion to `rocci_spec.md`. This document is **normative and self-contained**:
together with the spec, it fully determines the rocci implementation with no
reference to any `studroc_paper` source file. Routines are labeled A1–A16 and
cited from the spec by label.

Conventions used throughout:

- `neg`, `pos`: 1-D score arrays for the negative/positive class after
  ingestion; `n0 = len(neg)`, `n1 = len(pos)`.
- `grid`: the FPR evaluation grid, `linspace(0, 1, K)` with
  `K = min(512, n0 + 1)` unless the user overrides.
- `alpha = 1 - confidence`.
- All Python-side arithmetic is float64. The Rust kernel takes float64 scores
  and returns float64 TPRs.
- `Φ` / `Φ⁻¹` are the standard normal CDF / quantile (`scipy.stats.norm`).
- Pseudocode is executable NumPy unless marked otherwise; where a routine is
  marked **EXACT**, implement it as written (these encode validated tie/edge
  semantics that are easy to get subtly wrong).

Fidelity note: A1, A4–A9, A11 were transcribed from the validated
`studroc_paper` implementation; the golden-master fixtures (spec §5.7) are
recorded outputs of that implementation. If a golden test ever disagrees with
this appendix, the fixture wins and the appendix has a transcription bug —
fix the appendix, never regenerate the fixture to match new code.

---

## A1. Empirical ROC curve on a grid — **EXACT**

Right-continuous step interpolation of the empirical ROC, with `≥`-threshold
semantics on both classes. Ties in either class are handled by the
`searchsorted` calls; duplicated FPR vertices resolve to the largest TPR at
that FPR (the `side="right"` in the final lookup).

```python
def empirical_roc_on_grid(neg: np.ndarray, pos: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """TPR of the empirical ROC evaluated at each grid FPR (step convention)."""
    neg_asc = np.sort(neg)
    pos_asc = np.sort(pos)
    n0, n1 = len(neg_asc), len(pos_asc)

    thr_desc = neg_asc[::-1]                     # every negative is a threshold
    # counts with >= semantics: #{x >= v} = n - searchsorted_left(x_asc, v)
    fpr_v = (n0 - np.searchsorted(neg_asc, thr_desc, side="left")) / n0
    tpr_v = (n1 - np.searchsorted(pos_asc, thr_desc, side="left")) / n1

    fpr_v = np.concatenate(([0.0], fpr_v, [1.0]))
    tpr_v = np.concatenate(([0.0], tpr_v, [1.0]))

    idx = np.searchsorted(fpr_v, grid, side="right") - 1
    return tpr_v[np.clip(idx, 0, len(tpr_v) - 1)]
```

Complexity O(n log n). This replaces the paper repo's O(n²) broadcast; outputs
are identical (same vertex multiset, same step convention).

The full vertex list `(fpr_v, tpr_v)` (before grid interpolation) is also the
input to the AUC computation (A10) and to `RocBand.at()` (A13, same
`searchsorted(side="right") - 1` lookup).

---

## A2. Bootstrap TPR kernel (Rust) — **EXACT**

Computes the `(B, K)` matrix of bootstrap TPRs. Per replicate: resample both
classes with replacement; the TPR at grid point `t` is the fraction of
resampled positives **strictly greater** than the resampled negatives' order
statistic at descending 0-based index `k_t` (A14), where `k_t = n0` denotes a
−∞ sentinel (TPR = 1).

Algorithm: instead of sorting each resample (O(n log n)), tally draws into
count vectors over the pre-sorted originals and read thresholds/TPRs off with
linear walks — O(n0 + n1 + K) per replicate, O(n) memory per thread.

Reproducibility contract: the RNG stream is a pure function of
`(seed, replicate_index)`, so output is bit-identical regardless of thread
count or scheduling.

The following Rust is the kernel validated in the 2026-07-02 profiling session
(statistically indistinguishable from the paper implementation; see spec §9).
Wrap it in PyO3 (`rocci._core.bootstrap_tpr_matrix`) instead of the C ABI, and
switch score types to `f64`; the algorithm must not change.

```rust
#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

struct Xoshiro256pp { s: [u64; 4] }

impl Xoshiro256pp {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        Self { s: [splitmix64(&mut sm), splitmix64(&mut sm),
                   splitmix64(&mut sm), splitmix64(&mut sm)] }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]).rotate_left(23)
                              .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Lemire bounded sampling; bias < n * 2^-64 (negligible, accepted).
    #[inline(always)]
    fn next_bounded(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// One replicate. neg_sorted/pos_sorted ascending; k_indices ascending with
/// values in [0, n0]; cnt_* and thresholds are reused thread-local buffers.
fn replicate(
    rng: &mut Xoshiro256pp,
    neg_sorted: &[f64], pos_sorted: &[f64], k_indices: &[u64],
    cnt_neg: &mut [u32], cnt_pos: &mut [u32],
    thresholds: &mut [f64], out_row: &mut [f64],
) {
    let n_neg = neg_sorted.len();
    let n_pos = pos_sorted.len();
    let n_grid = k_indices.len();

    cnt_neg.fill(0);
    cnt_pos.fill(0);
    for _ in 0..n_neg { cnt_neg[rng.next_bounded(n_neg)] += 1; }
    for _ in 0..n_pos { cnt_pos[rng.next_bounded(n_pos)] += 1; }

    // Thresholds: walk the sorted negatives from the top. After consuming
    // value i (descending), `cum` resampled elements occupy descending
    // positions [0, cum); grid point j is resolved once cum > k_j.
    // Grid points with k == n_neg use the -inf sentinel (TPR = 1); they sit
    // at the tail of the ascending k_indices array.
    let n_resolved = k_indices.iter().take_while(|&&k| (k as usize) < n_neg).count();
    let mut j = 0usize;
    let mut cum: u64 = 0;
    let mut i = n_neg;
    while j < n_resolved && i > 0 {
        i -= 1;
        let c = cnt_neg[i];
        if c == 0 { continue; }
        cum += c as u64;
        while j < n_resolved && k_indices[j] < cum {
            thresholds[j] = neg_sorted[i];
            j += 1;
        }
    }
    debug_assert_eq!(j, n_resolved);

    // TPR: thresholds are non-increasing in j, so a single backward pointer
    // over the sorted positives accumulates counts strictly above each one.
    let mut p = n_pos;
    let mut acc: u64 = 0;
    let inv_n_pos = 1.0f64 / n_pos as f64;
    for j in 0..n_resolved {
        let thr = thresholds[j];
        while p > 0 && pos_sorted[p - 1] > thr {
            acc += cnt_pos[p - 1] as u64;
            p -= 1;
        }
        out_row[j] = acc as f64 * inv_n_pos;
    }
    for j in n_resolved..n_grid { out_row[j] = 1.0; }
}
```

Parallel driver (rayon): `out.par_chunks_mut(n_grid).enumerate()` with
`for_each_init` allocating the four thread-local buffers, and per replicate

```rust
let mut rng = Xoshiro256pp::new(seed ^ (rep as u64).wrapping_mul(0xA24BAED4963EE407));
```

If `n_threads > 0`, run inside `rayon::ThreadPoolBuilder::new()
.num_threads(n_threads).build()?.install(...)`; otherwise use the global pool.
Reject `n0 == 0 || n1 == 0 || K == 0 || B == 0` before entering Rust.

---

## A3. Bootstrap TPR kernel (NumPy fallback) — **EXACT**

Same statistical semantics as A2 (different RNG stream; cross-backend
agreement is distributional, spec §8.4). Vectorized over a batch of
replicates; only two O(K log n) `searchsorted` calls per replicate row run in
Python-level loops, which is negligible.

```python
def bootstrap_tpr_matrix_numpy(
    neg_sorted, pos_sorted, k_indices, n_boot, seed,
) -> np.ndarray:
    n0, n1, K = len(neg_sorted), len(pos_sorted), len(k_indices)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, K), dtype=np.float64)

    n_resolved = int(np.searchsorted(k_indices, n0, side="left"))  # k == n0 → sentinel
    out[:, n_resolved:] = 1.0
    ks = k_indices[:n_resolved]

    # cap batch memory at ~256 MB of int64 count matrices
    batch = max(1, min(n_boot, int(256e6 / (8 * (n0 + n1)))))
    p_neg = np.full(n0, 1.0 / n0)
    p_pos = np.full(n1, 1.0 / n1)

    for start in range(0, n_boot, batch):
        m = min(batch, n_boot - start)
        cnt_neg = rng.multinomial(n0, p_neg, size=m)      # (m, n0) counts
        cnt_pos = rng.multinomial(n1, p_pos, size=m)

        # descending cumulative counts over negatives:
        # cum[b, i] = # resampled negs among the (i+1) largest values
        cum_neg = np.cumsum(cnt_neg[:, ::-1], axis=1)
        # ascending cumulative counts over positives:
        cum_pos = np.cumsum(cnt_pos, axis=1)

        for b in range(m):
            # smallest i with cum_neg[b, i] > k  ⇒  threshold = (k+1)-th largest
            i = np.searchsorted(cum_neg[b], ks, side="right")
            thr = neg_sorted[::-1][i]                      # descending order values
            # strictly-greater count: n1_draws - #{resampled pos <= thr}
            pos_le = np.searchsorted(pos_sorted, thr, side="right")
            n_le = np.where(pos_le > 0, cum_pos[b][pos_le - 1], 0)
            out[start + b, :n_resolved] = (n1 - n_le) / n1
    return out
```

Emit `FallbackBackendWarning` once per process when this path is selected.

---

## A4. Wilson score machinery — **EXACT** (closed forms)

With `z > 0`, proportion array `p`, trials `n`:

```python
def wilson_bounds(p, n, z):
    """Two-sided Wilson interval, clipped to [0, 1]. n <= 0 → (0s, 1s)."""
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)

def wilson_halfwidth_sq(p, n, z):
    """Squared Wilson half-width; strictly positive even at p in {0, 1}.
    Used as the studentization variance floor: divide by z**2 to convert a
    half-width² at level z into a variance-scale quantity. n <= 0 → zeros."""
    denom = 1.0 + z * z / n
    return (z * z / denom**2) * (p * (1.0 - p) / n + z * z / (4.0 * n * n))
```

One-sided bounds at level `alpha` use `z1 = Φ⁻¹(1 - alpha)` and return only
`clip(center - half, 0, 1)` (lower) or `clip(center + half, 0, 1)` (upper)
with the same `center`/`half` formulas evaluated at `z1`.

Two-sided z at band level: `z_alpha = Φ⁻¹(1 - alpha / 2)`.

---

## A5. Wilson rectangle band (floor source) — **EXACT**

Pointwise 2-D confidence rectangles at each grid operating point; rocci uses
this **only** as the floor source inside A7, always with Šidák correction
across each rectangle's two margins and empirical TPR.

```python
def wilson_rectangle_band(neg, pos, grid, alpha):
    alpha_m = 1.0 - math.sqrt(1.0 - alpha)        # Šidák across 2 margins
    z = norm.ppf(1.0 - alpha_m / 2.0)
    n0, n1 = len(neg), len(pos)

    tpr = empirical_roc_on_grid(neg, pos, grid)   # A1
    fpr_lo, fpr_hi = wilson_bounds(grid, n0, z)   # A4 (uncertainty in FPR)
    tpr_lo, tpr_hi = wilson_bounds(tpr, n1, z)    # A4 (uncertainty in TPR)

    # upper envelope from upper-left corners (optimistic); lower from
    # lower-right corners (pessimistic)
    upper = _corner_envelope(x=fpr_lo, y=tpr_hi, grid=grid, take="max")
    lower = _corner_envelope(x=fpr_hi, y=tpr_lo, grid=grid, take="min")
    lower[0] = 0.0
    upper[-1] = 1.0
    return lower, upper

def _corner_envelope(x, y, grid, take):
    """Project corner points onto the grid: sort by x, collapse duplicate x
    (min for lower / max for upper), then linear interpolation with edge
    clamping (np.interp semantics)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    ux, inv = np.unique(xs, return_inverse=True)
    fill = np.inf if take == "min" else -np.inf
    uy = np.full(len(ux), fill)
    (np.minimum if take == "min" else np.maximum).at(uy, inv, ys)
    return np.interp(grid, ux, uy)
```

---

## A6. Studentized envelope — **EXACT**

Inputs: `boot_tpr` (B, K) from A2/A3, `tpr_hat` (K,) from A1, `alpha`,
`n0`, `n1`.

```python
z_alpha = norm.ppf(1.0 - alpha / 2.0)
var_raw = boot_tpr.var(axis=0, ddof=1)                    # keep: used by A7 gate
v_floor = wilson_halfwidth_sq(tpr_hat, n1, z_alpha) / z_alpha**2
sd = np.sqrt(np.maximum(var_raw, v_floor))

eps = min(1.0 / (n0 + n1), 1e-6)
dev = boot_tpr - tpr_hat                                  # (B, K), signed

# studentize with a collapse guard: where sd < eps, tiny deviations are
# numerical noise (score 0) and real deviations are scored against eps
stud = np.empty_like(dev)
ok = sd >= eps
stud[:, ok] = dev[:, ok] / sd[ok]
low = ~ok
stud[:, low] = np.where(np.abs(dev[:, low]) < eps, 0.0, dev[:, low] / eps)

ks = np.abs(stud).max(axis=1)                             # (B,) sup statistic
n_retain = math.ceil((1.0 - alpha) * n_boot)
threshold = np.sort(ks)[n_retain - 1]
retained = boot_tpr[ks <= threshold]                      # ties keep extra curves
lower_env = np.clip(retained.min(axis=0), 0.0, 1.0)
upper_env = np.clip(retained.max(axis=0), 0.0, 1.0)
```

`lower_env`/`upper_env` before any floor is the **pre-floor arm**, retained
for attribution (A9).

---

## A7. Variance-ratio gate + Wilson rectangle floor — **EXACT**

Detects where the bootstrap variance has collapsed below even the binomial
component and floors those points with A5 at a Šidák-corrected level. Note the
gate uses **raw** (unfloored) bootstrap variance.

```python
wilson_var = wilson_halfwidth_sq(tpr_hat, n1, z_alpha) / z_alpha**2
r = var_raw / np.clip(wilson_var, 1e-30, None)
deficiency = np.clip(1.0 - r, 0.0, None)          # (K,), >0 where collapsed
k_eff = float(deficiency.sum())
alpha_w = 1.0 - (1.0 - alpha) ** (1.0 / k_eff) if k_eff > 1.0 else alpha

lower, upper = lower_env.copy(), upper_env.copy()
needs = deficiency > 0
if needs.any():
    rect_lo, rect_hi = wilson_rectangle_band(neg, pos, grid, alpha_w)   # A5
    lower[needs] = np.minimum(lower[needs], rect_lo[needs])
    upper[needs] = np.maximum(upper[needs], rect_hi[needs])
    # enforce band monotonicity (ROC bands are non-decreasing):
    upper = np.maximum.accumulate(upper)
    lower = np.minimum.accumulate(lower[::-1])[::-1]
```

(The floor can only widen the band: it lowers `lower` and raises `upper`.)

---

## A8. Beta order-statistic floor — **EXACT**

Repairs the lower band at extreme FPR, where the dominant uncertainty is the
*horizontal* location of the threshold (an extreme order statistic of the
negatives) and no variance-based mechanism can see it. For continuous scores,
the true FPR exceedance at the j-th largest negative is exactly
`Beta(j, n0 + 1 - j)` regardless of the score distribution. `j_max = 25`,
fixed. Applied as a pointwise **minimum**: inside its jurisdiction it *lowers*
an overconfident envelope to the certified bound (this is an honesty repair —
it never raises the band); outside, `+inf` makes it a no-op.

```python
def beta_orderstat_floor(grid, lower, neg, pos, alpha, j_max=25):
    n0, n1 = len(neg), len(pos)
    j_used = min(j_max, n0)
    if j_used == 0 or n1 == 0:
        return lower
    a_e = alpha / (2 * j_max)                     # Bonferroni over 2*j_max one-sided events
    js = np.arange(1, j_used + 1)
    q = scipy.stats.beta.ppf(1.0 - a_e, js, n0 + 1 - js)   # jurisdiction edges, increasing

    # empirical TPR at the j-th largest negative, strictly-greater semantics
    neg_desc = np.sort(neg)[::-1][:j_used]
    pos_asc = np.sort(pos)
    tpr_hat_j = (n1 - np.searchsorted(pos_asc, neg_desc, side="right")) / n1

    # one-sided Wilson lower bounds (A4) at level a_e; prepend 0 so that
    # "no qualifying order statistic" (j_star == 0) maps to a vacuous floor
    bounds = np.concatenate(([0.0], wilson_lower_one_sided(tpr_hat_j, n1, a_e)))

    zone = (grid > 0.0) & (grid <= q[-1])
    if not zone.any():
        return lower
    floor = np.full_like(grid, np.inf)
    j_star = np.searchsorted(q, grid[zone], side="right")  # largest j with q_j <= t
    floor[zone] = bounds[j_star]
    return np.minimum(lower, floor)
```

Below `q[0]` the floor is 0 — vacuous by design; `vacuous_below = q[0]` on the
result object. With ties the true exceedance is stochastically smaller than
the Beta law, so discrete scores err conservative.

---

## A9. Assembly order and attribution — **EXACT** (order matters)

```python
lower_env, upper_env = A6(...)                    # pre-floor arm (keep a copy)
lower_rect, upper = A7(lower_env, upper_env)      # rectangle floor + monotonicity
lower = beta_orderstat_floor(grid, lower_rect, neg, pos, alpha)   # A8, last
lower[0] = 0.0                                    # pinned endpoints
lower[-1] = 1.0                                   # see delta note below
upper[-1] = 1.0
```

Do **not** re-run the monotonicity pass after A8 and do not reorder the
floors: the golden masters (spec §5.7) encode exactly this sequence.

Documented delta from the recorded implementation: the `lower[-1] = 1.0`
pin was added after fixture recording. The last grid point maps to the
`k = n0` sentinel, where the true ROC value is identically 1 — the pin
tightens the band at zero coverage cost. The golden-master test asserts it
separately and compares all other grid points against the untouched
fixtures (spec §5.6).

Attribution codes for the lower band (`tol = 1e-12`):

```python
attribution = np.zeros(K, dtype=np.int8)          # 0 = bootstrap envelope
attribution[lower_rect < lower_env - tol] = 2     # Wilson rectangle floor
attribution[lower < lower_rect - tol] = 1         # Beta floor (applied last, wins)
attribution[0] = 3                                # pinned endpoints
attribution[-1] = 3
```

---

## A10. AUC and bootstrap AUC CI

- `auc`: trapezoid rule over the full empirical vertex list from A1
  (`np.trapezoid(tpr_v, fpr_v)`) — not over the K-point grid.
- `auc_ci` (envelope path only): per-replicate trapezoid AUCs on the grid,
  `np.trapezoid(boot_tpr, grid, axis=1)`, then the empirical
  `(alpha/2, 1 - alpha/2)` percentiles (NumPy default `"linear"` method).
  Documented as a pointwise (scalar-parameter) percentile CI.

---

## A11. Working–Hotelling band (`normal=True`) — **EXACT**

Method-of-moments binormal fit with delta-method uncertainty, χ²₂ simultaneous
critical value, band built in probit space. Uses the same grid as §5.1.

```python
def working_hotelling_band(neg, pos, grid, alpha):
    n0, n1 = len(neg), len(pos)
    mu0, s0 = neg.mean(), neg.std(ddof=1)
    mu1, s1 = pos.mean(), pos.std(ddof=1)
    eps = 1e-8
    s0 = eps if (math.isnan(s0) or s0 < eps) else s0
    s1 = eps if (math.isnan(s1) or s1 < eps) else s1

    a = (mu1 - mu0) / s1                      # binormal intercept
    b = s0 / s1                               # binormal slope

    var_a = 1.0 / n1 + b * b / n0 + a * a / (2.0 * n1)
    var_b = b * b / (2.0 * n0) + b * b / (2.0 * n1)
    cov_ab = a * b / (2.0 * n1)
    if not math.isfinite(var_a):  var_a = 1.0 / n1
    if not math.isfinite(var_b):  var_b = 1.0 / n1
    if not math.isfinite(cov_ab): cov_ab = 0.0

    x = norm.ppf(np.clip(grid, 1e-9, 1.0 - 1e-9))          # probit FPR
    var_probit = var_a + x * x * var_b + 2.0 * x * cov_ab
    bad = ~np.isfinite(var_probit) | (var_probit < 0)
    var_probit[bad] = 1.0 / n1
    se = np.sqrt(var_probit)

    w = math.sqrt(scipy.stats.chi2.ppf(1.0 - alpha, df=2))  # simultaneous critical value
    center = a + b * x
    lower = norm.cdf(center - w * se)
    upper = norm.cdf(center + w * se)
    lower[0] = 0.0
    upper[-1] = 1.0
    return lower, upper
```

---

## A12. Normality diagnostics (with `normal=True`)

Plain-English is sufficient here; no exact code required.

1. Per class: Shapiro–Wilk if class size ≤ 5000, else D'Agostino K²
   (`scipy.stats.shapiro` / `normaltest`). Record test name, statistic, p.
2. Binormal-fit check: take the empirical vertex list from A1, keep vertices
   with both `fpr_v` and `tpr_v` in `(0.05, 0.95)`, deduplicate identical
   points, and require ≥ 10 remaining (else skip this check, `r2 = nan`).
   OLS of `Φ⁻¹(tpr_v)` on `Φ⁻¹(fpr_v)`; record R².
3. `suspect = (p_neg < 0.10) or (p_pos < 0.10) or (r2 is not nan and r2 < 0.98)`.
4. When `suspect`, emit `NormalityWarning` (spec §6 text); always attach the
   full `NormalityReport` to the result.

---

## A13. `RocBand.at()` interpolation

Step convention identical to A1's final lookup: for query FPR values `t`,
`idx = clip(searchsorted(grid, t, side="right") - 1, 0, K - 1)` and return
`(lower[idx], tpr[idx], upper[idx])`. Queries outside `[0, 1]` raise
`RocciError`.

---

## A14. Grid → order-statistic index mapping — **EXACT**

```python
k_indices = np.clip(np.floor(grid * n0), 0, n0).astype(np.uint64)
```

`k_indices` is non-decreasing because `grid` is. `k = n0` (which occurs only
at `t = 1.0`) is the −∞ sentinel: TPR = 1 for that replicate/point. This
mapping plus A2's strictly-greater positive count fully defines the bootstrap
TPR semantics.

---

## A15. Calibration-gate DGPs and coverage criterion (tests only)

Four data-generating processes with computable true ROCs. Constants (the
shift `d` solving `AUC = 0.8` for each continuous DGP) are solved once
offline by root-finding on `AUC(d) = P(pos > neg)` (numeric integration) and
hardcoded in the test file with the solver kept alongside in a comment.

1. **Binormal**: `neg ~ N(0,1)`, `pos ~ N(d,1)` with `d = √2 · Φ⁻¹(0.8)`.
   True ROC: `R(t) = Φ(d + Φ⁻¹(t))`.
2. **Student-t (df=3)**: `neg ~ t₃`, `pos ~ t₃ + d`.
   True ROC: `R(t) = 1 − T₃(T₃⁻¹(1−t) − d)` with `T₃` the t₃ CDF.
3. **Bimodal negatives**: `neg ~ ½·N(−1, 0.5²) + ½·N(1, 0.5²)`,
   `pos ~ N(d, 0.75²)`. True ROC: `R(t) = 1 − G(F⁻¹(1−t))` where `F` is the
   mixture CDF (closed form), `F⁻¹` by `scipy.optimize.brentq`, and `G` the
   positive-class normal CDF.
4. **Discretized binormal (ties cell)**: draw from DGP 1, then round every
   score to the nearest `h = 0.1` (typically ~60 distinct values). This cell
   exists to verify *empirically* that ties leave the band conservative
   (spec §4.3) rather than merely argued-safe. The population ROC of the
   rounded scores is a step function, computed exactly:

   - Support: `v_i = i · h` for integers `i` covering ±5σ of both classes.
   - `P(round(X) ≥ v) = P(X ≥ v − h/2) = 1 − Φ((v − h/2 − μ)/σ)` per class.
   - Vertices: `(P(neg_r ≥ v), P(pos_r ≥ v))` over the support in descending
     `v` order, plus (0,0) and (1,1) — the population analogue of A1's vertex
     list. Evaluate `R(t_k)` at grid points with the same
     `searchsorted(side="right") − 1` step lookup as A1.

   Note the true AUC of this DGP differs slightly from 0.8 after rounding;
   that is irrelevant — coverage is scored against this cell's own exact
   step-function ROC.

Per simulation: draw `(neg, pos)` at the configured n with a fixed seed
sequence (`seed = hash((dgp, n, sim_index))` via `np.random.SeedSequence`),
run `roc_band` with a fixed `random_state`, and score:

- **covered** iff `lower[k] ≤ R(t_k) ≤ upper[k]` for **all** grid points k;
- **width** = mean of `upper − lower` over the grid.

Gate assertions per (DGP, n) cell over 250 sims at confidence 0.95: empirical
coverage within the two-sided binomial(250, 0.95) central 99.9% interval
(precomputed constants ≈ [0.906, 0.988] — deterministic given fixed seeds,
the interval only documents the tolerance's origin), and mean width strictly
below the A16 KS band's mean width on the same draws.

---

## A16. KS/DKW fixed-width reference band (tests only)

Used solely as the width yardstick in A15 — never in the public API. This
definition is normative for the tests:

```python
alpha_m = 1.0 - math.sqrt(1.0 - alpha)        # Šidák across the two ECDFs
c = math.sqrt(math.log(2.0 / alpha_m) / 2.0)  # DKW critical value
d0, d1 = c / math.sqrt(n0), c / math.sqrt(n1) # horizontal, vertical margins

# R_hat = A1 step function evaluated via empirical_roc_on_grid
upper = np.clip(empirical_roc_on_grid(neg, pos, np.clip(grid + d0, 0, 1)) + d1, 0, 1)
lower = np.clip(empirical_roc_on_grid(neg, pos, np.clip(grid - d0, 0, 1)) - d1, 0, 1)
lower[0] = 0.0
upper[-1] = 1.0
```
