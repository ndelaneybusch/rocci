# rocci — Package Specification

**Distribution-free simultaneous confidence bands for ROC curves.**

This document is the complete build specification for `rocci`, a production-quality
package implementing the studentized bootstrap envelope method validated in
`studroc_paper`. It contains every decision needed to build the package from an
empty repository. Statistical semantics are defined **normatively** in the
companion appendix `rocci_spec_appendix.md` (labeled routines A1–A16, cited
from here by label); the two files travel together and no `studroc_paper`
source file is required to build rocci. The only artifacts imported from the
paper repo are *data*: the golden-master fixtures (§5.7) and figure/content
material for the docs.

---

## 1. Locked decisions


| Decision                               | Choice                                                    | Rationale                                                                                                                                       |
| -------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Name (PyPI, conda-forge, import, repo) | `rocci`                                                   | Owner's choice; short, evokes "ROC confidence intervals"                                                                                        |
| Compute backend                        | **Pure Rust** (PyO3/maturin), NumPy fallback              | Benchmarked: Rust beats RTX 3080 CUDA 1.7–2.0× at every size and handles n=10M where torch OOMs at n=1M (§9). No GPU router, no torch anywhere. |
| Python floor                           | 3.10+                                                     | Reach; abi3-py310 wheels cover all newer interpreters with one build                                                                            |
| API shape                              | `roc_band()` → `RocBand` result object                    | scipy/statsmodels idiom; lowest friction                                                                                                        |
| Core runtime deps                      | `numpy>=1.24`, `scipy>=1.10` — nothing else               | "Just works and is fast" requires a tiny dependency footprint                                                                                   |
| Plotting                               | `matplotlib` via optional extra `rocci[plot]`             | Lazy import with actionable error message                                                                                                       |
| License                                | MIT                                                       | Maximize adoption (owner may override)                                                                                                          |
| Versioning                             | SemVer, start at `0.1.0`, `1.0.0` when out of beta        |                                                                                                                                                 |

---

## 2. Product definition

### 2.1 What v1.0 does

- `roc_band(y_true, y_score, confidence=0.95, n_boot=2000, normal=False, ...)`
returns a simultaneous confidence band for the true population ROC curve.
- `normal=False` (default): the **studentized bootstrap envelope** with the
Wilson rectangle floor and exact Beta order-statistic floor — distribution-free,
calibrated across score distributions (the paper's method).
- `normal=True`: the **Working–Hotelling** binormal band, always accompanied by
normality diagnostics and loud warnings when binormality looks doubtful.
- Ingestion of scores/labels from the ecosystem's common containers without
requiring any of those libraries (§4).
- Multiclass via `roc_band_ovr` (§3.6): one-vs-rest bands with an exact
(conservative) family-wise coverage guarantee across classes.
- Paper-quality plot of the band; floor-attribution diagnostic plot.
- Fast by default: Rust core wheels for all mainstream platforms; a pure-NumPy  
fallback keeps the package functional everywhere else.

---

## 3. Public API

### 3.1 Top-level surface (`rocci/__init__.py`)

```python
from rocci import roc_band, roc_band_ovr, from_estimator, RocBand, __version__, show_versions
```

`show_versions()` prints an environment report for bug reports: rocci version,
active backend (rust/numpy), numpy/scipy/matplotlib versions, OS, CPU count.
Everything else is private (`rocci._*` or documented as internal).

### 3.2 `roc_band`

```python
def roc_band(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    normal: bool = False,
    grid_size: int | None = None,
    pos_label: int | str | bool | None = None,
    score_reduce: Literal["mean", "median"] | None = None,
    nan_policy: Literal["raise", "omit"] = "raise",
    random_state: int | None = None,
    diagnostics: bool = False,
    n_threads: int | None = None,
) -> RocBand
```


| Parameter      | Semantics                                                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `y_true`       | Labels in any form §4 can coerce.                                                                                                                                               |
| `y_score`      | Scores in any form §4 can coerce (higher = more positive).                                                                                                                      |
| `confidence`   | Simultaneous coverage target, in (0, 1). `alpha = 1 - confidence`.                                                                                                              |
| `n_boot`       | Bootstrap replicates. `ValueError` if `< 100`; warn if `< 1000` (quantile resolution). Ignored with `normal=True`.                                                              |
| `normal`       | `False` → envelope method (§5). `True` → Working–Hotelling (§6) plus normality diagnostics.                                                                                     |
| `grid_size`    | FPR grid points `K`. Default `None` → `K = min(512, n_neg + 1)`. Grid is `linspace(0, 1, K)`.                                                                                   |
| `pos_label`    | Which label of `y_true` is positive. Default: infer (§4.2).                                                                                                                     |
| `score_reduce` | Only for 2-D posterior-draw score inputs (§4.3). Default `None` errors on ambiguous 2-D input other than `(n, 2)` probabilities.                                                |
| `nan_policy`   | `"raise"` (default) or `"omit"` (drop rows with NaN in either input, warn with count).                                                                                          |
| `random_state` | Seeds the bootstrap. Same seed + same backend + same version ⇒ bit-identical band. Cross-backend (Rust vs fallback) results agree statistically, not bit-wise — documented.     |
| `diagnostics`  | If `True`, render the floor-attribution diagnostic figure (§7.3) immediately (notebook convenience). Attribution data is *always* computed and stored on the result regardless. |
| `n_threads`    | Rust thread count; `None` → all cores.                                                                                                                                          |


### 3.3 `from_estimator`

```python
def from_estimator(estimator, X, y, *, response_method="auto", **roc_band_kwargs) -> RocBand
```

Duck-typed sklearn convenience (mirrors `RocCurveDisplay.from_estimator`):
uses `predict_proba(X)[:, pos_idx]` if available, else `decision_function(X)`.
No sklearn import; works with anything exposing those methods.

### 3.4 `RocBand` result object

Frozen `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True)
class RocBand:
    fpr: np.ndarray            # (K,) grid
    tpr: np.ndarray            # (K,) empirical ROC at grid
    lower: np.ndarray          # (K,) lower band
    upper: np.ndarray          # (K,) upper band
    confidence: float
    method: Literal["envelope", "working_hotelling"]
    n_neg: int
    n_pos: int
    n_boot: int | None         # None for WH
    auc: float                 # trapezoid on the full empirical ROC (not the grid)
    auc_ci: tuple[float, float] | None   # percentile bootstrap CI (envelope path only)
    attribution: np.ndarray    # (K,) int8 codes: 0=bootstrap, 1=beta floor, 2=wilson floor, 3=pinned endpoint (envelope path; zeros for WH)
    vacuous_below: float | None  # FPR below which the lower band is provably vacuous (~q_1 = Beta(1, n_neg).ppf(1 - alpha/(2*j_max))); None for WH
    normality: NormalityReport | None    # populated when normal=True
    backend: Literal["rust", "numpy"]
    random_state: int | None
```

Methods:

- `at(fpr: ArrayLike) -> tuple[lower, tpr, upper]` — step-interpolated values at
arbitrary FPR points (same step convention as the band construction;
appendix A13).
- `plot(ax=None, **style) -> matplotlib.axes.Axes` — §7.2.
- `plot_diagnostics(ax=None) -> matplotlib.figure.Figure` — §7.3.
- `summary() -> str` — human-readable report: sample sizes, AUC and band area,
coverage level, backend, the vacuous-region boundary ("no distribution-free
lower bound exists below FPR ≈ x; increase the number of negatives to certify
lower FPRs"), floor jurisdictions, and (WH path) the normality verdict.
- `to_dataframe()` — pandas DataFrame `[fpr, lower, tpr, upper, attribution]`;
lazy pandas import, actionable error if absent.
- `band_area: float` property — mean vertical width (the paper's tightness metric).

`NormalityReport` (frozen dataclass): per-class test name, statistic, p-value;
probit–probit linearity R²; boolean `suspect`; the exact warning text emitted.

### 3.5 Warnings and exceptions (`rocci._warnings`, `rocci._exceptions`)

- `RocciWarning(UserWarning)` base; subclasses `NormalityWarning`,
`SmallSampleWarning`, `TiesWarning`, `FallbackBackendWarning` (once per process
when the Rust core is missing).
- `RocciError(ValueError)` base for all input errors, with messages that state
the fix, not just the failure.

### 3.6 `roc_band_ovr` (multiclass)

```python
def roc_band_ovr(
    y_true: ArrayLike,              # m > 2 distinct labels
    y_score: ArrayLike,             # (n, m) score/probability matrix
    *,
    confidence: float = 0.95,
    family: Literal["bonferroni", "none"] = "bonferroni",
    classes: Sequence | None = None,
    **roc_band_kwargs,
) -> dict[Any, RocBand]
```

One-vs-rest reduction, a thin loop over `roc_band` (~40 lines; no new
statistics beyond the alpha split):

- Column `j` of `y_score` scores class `classes[j]` against the rest;
  `classes` defaults to `np.unique(y_true)` order (matching sklearn's
  `predict_proba` convention). Mismatched m vs. column count →
  `RocciError`.
- `family="bonferroni"` (default): each class band is built at confidence
  `1 − α/m` with `α = 1 − confidence`. Bonferroni requires no independence,
  so joint coverage of **all m** one-vs-rest curves is ≥ `confidence` —
  exact (conservative) family-wise validity, not an approximation.
  `family="none"`: each band at `confidence` marginally, no joint claim.
- Per-class bootstraps are seeded independently and reproducibly via
  `np.random.SeedSequence(random_state).spawn(m)`.
- `normal=True` **raises `RocciError`**: the one-vs-rest "rest" class is by
  construction a mixture of the remaining classes — structurally the
  bimodal-negatives regime where Working–Hotelling coverage collapses
  (paper: ~0.23 and worsening with n). The error text explains this and
  notes that users who insist can loop `roc_band(normal=True)` per class
  themselves; rocci will not automate a construction that manufactures the
  parametric band's proven worst case.
- Each returned `RocBand` carries its per-class effective confidence in
  `confidence`; `summary()` on any of them states the family context is the
  caller's responsibility to report.

Docs callout (§13): the envelope *composes* to multiclass (distribution-free
per curve + Bonferroni = clean family guarantee), while WH *anti-composes*
(OvR creates exactly its failure mode). This asymmetry is a selling point.

---

## 4. Ingestion (`rocci/ingest.py`)

Goal: accept what users actually have, with **zero hard dependencies** on the
source libraries. All coercion is duck-typed / protocol-based.

### 4.1 Container coercion (applies to both inputs)

Ordered attempts:

1. `numpy.ndarray` → as-is.
2. Objects implementing `__dlpack__` (torch/JAX/CuPy tensors) → try
  `np.from_dlpack`; on failure (e.g. CUDA tensor), duck-call
   `.detach().cpu().numpy()` if those attributes exist, else raise
   with a message naming the fix.
3. Objects implementing `__array__` (pandas Series/Index, polars Series via
  `.to_numpy()` duck-call, arviz/xarray DataArray) → `np.asarray`.
4. Python sequences → `np.asarray`.

Result must be numeric (or bool/str for labels), finite checks per `nan_policy`
(±inf allowed in scores — legal and handled by sorting; NaN is not).

### 4.2 Labels `y_true`

- Accept bool, `{0,1}`, `{-1,1}`, or exactly two distinct values of any dtype
(incl. strings) with `pos_label` resolving which is positive.
- Inference when `pos_label is None`: bool → `True`; `{0,1}` → `1`;
`{-1,1}` → `1`; otherwise `RocciError` demanding explicit `pos_label`.
- **More than two distinct labels → `RocciError`** naming the fix: "there is
no single ROC curve for m > 2 classes; use `roc_band_ovr` for one-vs-rest
bands with a family-wise guarantee, or binarize and pass `pos_label`."
`roc_band` never silently picks a multiclass reduction.
- `RocciError` if only one class present, or if `n_neg < 2` / `n_pos < 2`.
- `SmallSampleWarning` if `n_neg < 20` or `n_pos < 20` (band will be dominated
by the exact floors; still valid).

### 4.3 Scores `y_score`

- 1-D numeric: used directly.
- 2-D `(n, 2)` with rows summing to ≈1 (tolerance 1e-6, checked on a sample):
treated as `predict_proba` output → take positive-class column (column of
`pos_label` position; column 1 by default), emit an INFO-level note in
`summary()`, not a warning.
- 2-D `(draws, n)` or 3-D `(chain, draw, n)` (PyMC/arviz posterior predictive
probabilities): require `score_reduce` to be set (`"mean"` or `"median"`);
reduce over draw axes; `RocciError` naming `score_reduce` if unset. Note in
docs: the band then quantifies sampling uncertainty of the reduced scores,
not posterior uncertainty.
- Ties/discreteness: **never a failure** — every layer of the envelope is
either tie-indifferent (empirical ROC, bootstrap kernel, Wilson floors) or
conservative under ties (Beta floor: the true exceedance is stochastically
smaller than the Beta law, so discrete scores err safe). If
`n_unique / n < 0.5`, emit `TiesWarning` saying exactly that; the claim is
verified empirically by the discretized-DGP calibration cell (appendix
A15.4). Degenerate extreme — a class with constant scores
(`n_unique == 1`) — still proceeds, with `TiesWarning` text noting the band
degenerates to the exact floors and will be honestly wide.
- Probability-vs-logit note in docs: the envelope band is invariant to any
strictly monotone transform of scores (rank-based bootstrap), unlike WH —
this is a selling point and gets a docs callout.

---

## 5. Statistical specification — envelope path (`normal=False`)

The single validated configuration from the paper (probability-space
construction, KS retention, empirical TPR, Wilson + Beta floors). Normative
algorithms: appendix A1–A10; this section states what each stage is *for* and
which routine defines it.

### 5.1 Grid and empirical curve

- `K = grid_size or min(512, n_neg + 1)`; grid `t_k = linspace(0, 1, K)`.
- Empirical ROC on the grid: **appendix A1** (right-continuous step
interpolation, `≥`-threshold tie semantics). A1 is O(n log n), replacing the
paper repo's O(n²) broadcast (measured at ~6 s for n=100k; must be < 20 ms
here).

### 5.2 Bootstrap kernel (Rust; §8)

For each replicate `b = 1..B`, resample n_neg negatives and n_pos positives
with replacement; the TPR at grid point `t` equals the fraction of resampled
positives **strictly greater** than the resampled negatives' order statistic at
descending 0-based index `k_t` (**appendix A14**), where `k_t = n_neg` denotes
a −∞ sentinel (TPR = 1). The counting algorithm and RNG are **appendix A2**
(Rust, exact code) / **A3** (NumPy fallback). These semantics were validated
statistically against the paper's torch kernel (max mean-difference z = 2.47
at B = 8000).

### 5.3 Studentization, retention, envelope

Normative: **appendix A6** (Wilson formulas: **A4**). In brief: deviations
from `R̂` are studentized by the bootstrap SD floored at the Wilson variance
(`z_α = Φ⁻¹(1 − α/2)`), with the `ε = min(1/(n_neg+n_pos), 1e-6)` collapse
guard; each replicate is scored by its supremum absolute studentized
deviation; the `ceil((1−α)·B)` most typical replicates (ties at the threshold
included) are retained; the band is their pointwise min/max clipped to [0, 1].

### 5.4 Wilson rectangle floor (variance-ratio gated)

Normative: **appendix A7** (rectangle sub-band: **A5**). Deficiency
`w(t) = max(0, 1 − v_raw(t)/v_wilson(t))` — note *raw*, unfloored bootstrap
variance — `K_eff = Σ w(t)`, Šidák `α_w = 1 − (1−α)^{1/K_eff}` (if
`K_eff > 1`), the A5 Wilson rectangle band at `α_w` applied as a floor
wherever `w(t) > 0`, then band monotonicity enforced (upper: running max
left→right; lower: running min right→left).

### 5.5 Beta order-statistic floor (lower band, low FPR)

Normative: **appendix A8**, with `j_max = 25` (fixed, not exposed): per-event
level `α_e = α/(2·j_max)`; `q_j = Beta(j, n_neg+1−j).ppf(1−α_e)`; floor at
grid t = one-sided Wilson lower bound (level `α_e`) of the empirical TPR at
the largest j with `q_j ≤ t`; vacuous (0) below `q_1`. Applied as pointwise
**min** with the current lower band within its jurisdiction.

### 5.6 Assembly and attribution

- Pipeline order (envelope → rectangle floor + monotonicity → Beta floor →
pinned endpoints) and the attribution rule are **appendix A9** — the order
is load-bearing and must not be changed.
- Pinned endpoints are `lower[0] = 0`, `upper[-1] = 1`, **and**
`lower[-1] = 1`. The last pin is a documented statistical delta from the
recorded paper implementation (which left `lower[-1]` at the floored value):
at FPR = 1 the true ROC is identically 1, so pinning tightens the band at
zero coverage cost. The golden-master test therefore compares `lower[:-1]`
against the fixtures (which are **not** regenerated) and asserts the pin
separately.
- `attribution[k]`: which mechanism produced the final `lower[k]` —
bootstrap envelope, Beta floor, Wilson floor, or pinned endpoint — from the
A9 comparison against the retained pre-floor envelope arm.
- `auc` and `auc_ci`: **appendix A10** (trapezoid point estimate on the full
vertex list; percentile CI of per-replicate grid AUCs).
- `vacuous_below = q_1` (from A8).

### 5.7 Golden-master equivalence requirement

Given identical `(boot_tpr_matrix, fpr_grid, y_true, y_score, alpha)` fixtures
— recorded once from the validated paper implementation and committed as data
— the new assembly (§5.3–5.6, appendix A4–A9) must reproduce the recorded
band outputs within `atol=1e-6` (float32 fixtures widened to float64). This
is a CI test with committed fixture files covering: small n (30/30),
unbalanced (50/500), heavy ties, high AUC (0.95), and n=10k. Precedence rule
(also stated in the appendix header): if a golden test disagrees with an
appendix routine, the fixture wins — the appendix has a transcription bug;
never regenerate fixtures to match new code.

---

## 6. Statistical specification — Working–Hotelling path (`normal=True`)

- Working–Hotelling band: **appendix A11** (exact) — method-of-moments
binormal fit (`a = (μ₁−μ₀)/s₁`, `b = s₀/s₁`), delta-method covariance, χ²₂
critical value, band in probit space, mapped back; degenerate-case guards
as written in A11.
- **Normality diagnostics always run** (`rocci/band/normal.py`,
**appendix A12**):
  - Per class: Shapiro–Wilk for `n ≤ 5000`, else D'Agostino K².
  - Binormal-fit check: OLS R² of `Φ⁻¹(TPR)` vs `Φ⁻¹(FPR)` over the empirical
  ROC interior (drop the 5% tails; A12 gives the vertex-selection rule).
  - `suspect = (either class p < 0.10) or (R² < 0.98)`.
- When `suspect`, emit `NormalityWarning` whose text states the paper's finding
verbatim in spirit: WH coverage degrades *continuously* with departures from
binormality and worsens with n — there is no safe diagnostic region — and
recommends `normal=False`. If the §4.3 ties threshold also fired, the warning
appends: "scores contain heavy ties, which are incompatible with the binormal
model (ties have probability zero under it)." The full `NormalityReport` is
attached to the result either way.
- Multiclass note: `roc_band_ovr` refuses `normal=True` outright (§3.6) — the
one-vs-rest rest-class is a mixture, WH's proven failure mode.
- `n_boot`, `random_state` ignored (documented); `attribution` all zeros;
`auc_ci = None`; `vacuous_below = None`.

---

## 7. Visualization (`rocci/plotting.py`, extra `rocci[plot]`)

### 7.1 General

- Lazy `import matplotlib`; absence raises
`RocciError("plotting requires matplotlib — pip install 'rocci[plot]'")`.
- House style: colorblind-safe palette, no chartjunk, serif-neutral, vector-
friendly (all elements legend-labeled; works in PDF/SVG). Default figsize
(5.5, 5.0), constrained layout. Every plot returns the Axes/Figure for
composition and takes `ax=`.

### 7.2 `RocBand.plot()`

Single panel: shaded band (`fill_between`, alpha≈0.25), empirical ROC step
line, chance diagonal (dotted), optional `show_vacuous=True` hatching of the
region below `vacuous_below` on the FPR axis, annotation
`"{confidence:.0%} simultaneous band (rocci envelope)"` or `"(Working–Hotelling)"`.
Style overrides via kwargs (`color`, `band_alpha`, `label`, ...).

### 7.3 `RocBand.plot_diagnostics()`

Two-panel figure mirroring the paper's floor-attribution graphics
(fig15a / lower-band waterfall):

1. Band with the lower bound color-coded by `attribution` (unshaded interior =
  bootstrap; yellow = Beta order-statistic floor; green = Wilson rectangle
   floor), with jurisdiction boundaries marked.
2. Variance channels vs FPR (log y): raw bootstrap variance, Wilson variance
  floor, and shaded intervals where each floor is active — the "why did my
   band do that here" view.

For the WH path, panel 1 shows the band and panel 2 shows the two normality
QQ plots + probit–probit ROC linearity with R², replacing floor attribution.

---

## 8. Rust core and backend routing

### 8.1 Crate (`rust/` in the rocci repo)

- Mixed maturin layout: `pyproject.toml` with `[build-system] maturin`,
Python in `python/rocci/`, crate in `rust/` compiled to module `rocci._core`.
- PyO3 (abi3-py310) + rayon + numpy crate. RNG hand-rolled xoshiro256++ seeded
per replicate via `splitmix64(seed ⊕ rep·0xA24BAED4963EE407)` — RNG streams
are replicate-indexed, so results are **independent of thread count and
scheduling** (hard requirement; tested).
- Exported function (single entry point; keep the Rust surface minimal):

```rust
fn bootstrap_tpr_matrix(
    neg_sorted: &[f64], pos_sorted: &[f64],
    k_indices: &[u64],           // ascending, values in [0, n_neg]
    n_boot: usize, seed: u64, n_threads: usize,
) -> ndarray (n_boot, K) f64
```

- Algorithm: **appendix A2** (exact Rust, validated in the profiling
prototype): per replicate, tally resample counts into `u32` vectors over
the pre-sorted scores; walk the negative counts top-down to resolve grid
thresholds; two-pointer walk over positive counts for strictly-greater TPR
counts. O(n_neg + n_pos + K) per replicate; O(n) memory per thread.
- Everything else (studentization, retention, floors, WH) stays in
NumPy/SciPy — it is O(B·K) and negligible.
- Rust QA: `cargo test` (unit tests incl. exactness against a brute-force
sort-based oracle on small inputs), `cargo clippy -D warnings`, `cargo fmt`.

### 8.2 NumPy fallback (`rocci/backend/_fallback.py`)

Same statistical semantics, vectorized NumPy: **appendix A3** (exact) —
batched `Generator.multinomial` count matrices, `cumsum` + `searchsorted` for
thresholds and TPR counts, memory-capped batching. Expected 10–30× slower
than Rust — still far faster than sort-based bootstrapping. Emits
`FallbackBackendWarning` once per process.

### 8.3 Selection

```python
try:
    from rocci import _core          # Rust
    BACKEND = "rust"
except ImportError:
    from rocci.backend import _fallback
    BACKEND = "numpy"
```

Env override `ROCCI_BACKEND={rust,numpy}` for testing/debugging only
(undocumented in user docs, documented in CONTRIBUTING). No other routing —
performance is invisible to users.

### 8.4 Cross-backend contract test

Rust and fallback given the same seed produce different streams (documented),
but a CI test asserts distributional agreement exactly as the profiling sanity
check did: max pointwise mean-difference z < 6 at B = 8000 and interior std
ratios within [0.9, 1.1].

---

## 9. Performance requirements

Baseline measurements from the routing decision:


| n_total | B    | torch CUDA | torch CPU | Rust prototype |
| ------- | ---- | ---------- | --------- | -------------- |
| 1k      | 1000 | 0.002 s    | 0.023 s   | 0.001 s        |
| 10k     | 4000 | 0.020 s    | 0.728 s   | 0.011 s        |
| 100k    | 4000 | 0.180 s    | 8.21 s    | 0.089 s        |
| 1M      | 4000 | OOM        | OOM       | 2.55 s         |
| 10M     | 1000 | —          | —         | 35.4 s         |


Release gates (12-core desktop class, defaults `n_boot=2000`, K=512):

- end-to-end `roc_band` at n=10k: **< 100 ms**; n=100k: **< 500 ms**;
n=1M: **< 3 s**.
- Non-bootstrap overhead (ingest + empirical ROC + floors + assembly) at
n=100k: **< 50 ms** (kills the O(n²) empirical-ROC pattern by construction).
- Memory: no allocation proportional to `n_boot × n`; peak extra memory
≤ `B·K·8` bytes + O(n) per thread.
- `benchmarks/` keeps `asv`-style scripts (plain `time.perf_counter` is fine)
with these gates as regression thresholds in the required per-PR `perf` job
(§14.2), measured relative to main to absorb runner variance.

---

## 10. Repository layout

```
rocci/
├── pyproject.toml            # maturin build backend; abi3-py310
├── Cargo.toml                # workspace → rust/
├── rust-toolchain.toml       # pinned stable Rust + components (rustfmt, clippy)
├── justfile                  # canonical dev commands (§12.2)
├── .pre-commit-config.yaml   # §12.3
├── CONTRIBUTING.md           # §12.5
├── CITATION.cff
├── CHANGELOG.md              # generated by git-cliff (§14.5)
├── rust/
│   ├── Cargo.toml
│   └── src/lib.rs            # PyO3 module rocci._core
├── python/rocci/
│   ├── __init__.py           # roc_band, roc_band_ovr, from_estimator, RocBand, __version__
│   ├── _api.py               # orchestration of §5/§6 pipelines
│   ├── _result.py            # RocBand, NormalityReport
│   ├── _validation.py        # shared arg checks
│   ├── _warnings.py, _exceptions.py
│   ├── ingest.py             # §4
│   ├── band/
│   │   ├── envelope.py       # §5.3–5.6
│   │   ├── floors.py         # Wilson rectangle + Beta order-stat floors
│   │   ├── grids.py          # grid rule, empirical ROC (O(n log n))
│   │   └── normal.py         # §6 WH + diagnostics
│   ├── backend/
│   │   ├── __init__.py       # selection (§8.3)
│   │   └── _fallback.py      # §8.2
│   └── plotting.py           # §7
├── tests/                    # pytest; fixtures/ holds golden masters (§5.7)
├── benchmarks/
├── docs/                     # mkdocs-material site source (§13)
│   └── vignettes/            # jupytext .md notebooks, executed at docs build (§13.3)
└── .github/
    ├── workflows/            # ci.yml, gates.yml, wheels.yml, docs.yml (§14)
    ├── ISSUE_TEMPLATE/       # bug (asks for rocci.show_versions() output), feature, docs
    ├── PULL_REQUEST_TEMPLATE.md
    ├── dependabot.yml        # actions + cargo + pip ecosystems, weekly
    └── CODEOWNERS
```

`pyproject.toml` essentials: `requires-python = ">=3.10"`; deps `numpy`,
`scipy`; extras `plot = ["matplotlib>=3.7"]`,
`dev = [pytest, pytest-cov, hypothesis, ruff, ty, maturin, pandas, matplotlib]`;
ruff + ty configured per §12.4.

---

## 11. Testing plan

Style per the owner's standing test guidance: red-team the behavior, every
test mitigates a named risk, parametrize, realistic data, minimal mocking at
low levels / routing-focused mocking at the API level.

1. **Golden-master equivalence** (§5.7) — the rewrite cannot silently change
  the validated statistics. Highest-value suite; write it first.
2. **Rust kernel oracle tests** (in Rust): counting kernel ≡ brute-force
  sort-based bootstrap for n ≤ 64 across seeds, including all-ties inputs,
   k=0 and k=n_neg sentinels, n_neg=1.
3. **Determinism**: same seed ⇒ bit-identical band across `n_threads` ∈
  {1, 4, all}; across two runs; fallback deterministic under seed too.
4. **Cross-backend distributional agreement** (§8.4).
5. **Statistical calibration (required PR merge gate, §14.2)**: four DGPs
   (binormal, Student-t df=3, bimodal negatives, discretized binormal — the
   ties cell that turns "conservative under ties" from argument into
   evidence) × n ∈ {30, 200} × 250 sims with **fixed seeds** (deterministic —
   a regression test of the assembled method, not a stochastic gate):
   envelope coverage within precomputed Monte-Carlo bands around nominal and
   mean width < the KS reference band's width. DGPs, true-ROC formulas,
   seeding, and the coverage criterion: **appendix A15**; the KS yardstick
   band: **appendix A16**. Guards the assembled method, not just components.
   (Calibration target: coverage *near* nominal from above — closer is
   better, not higher.)
6. **Ingestion matrix**: parametrized over containers (lists, ndarray, pandas,
  polars-mock via `.to_numpy` duck object, torch-mock via
   `__dlpack_`_/`.detach().cpu().numpy()` stub, (n,2) proba, (draws,n) with
   and without `score_reduce`, string labels ± `pos_label`, NaN under both
   policies, ±inf scores, single-class error, tiny classes, >2 labels →
   `RocciError` naming `roc_band_ovr`, constant-score class → proceeds with
   `TiesWarning` and floors-only band).
6b. **`roc_band_ovr` contract**: per-class confidence equals `1 − α/m` under
   `family="bonferroni"` (checked via the returned `confidence` fields);
   column↔label routing matches `classes` order against a hand-built 3-class
   fixture; per-class seeds differ but are reproducible from `random_state`;
   `normal=True` raises `RocciError` mentioning the rest-class mixture; a
   3-class calibration spot-check (binormal DGP, one cell) verifies joint
   OvR coverage ≥ family confidence with fixed seeds.
7. **API contract**: warning taxonomy fires exactly when specified
  (`NormalityWarning` on a t₃ dataset with `normal=True`; none on binormal);
   `confidence=0.5` warns; `n_boot=50` raises; result immutability; `at()`
   step-consistency with band arrays; monotone `lower`/`upper`; `lower ≤ tpr ≤  upper` everywhere except documented pinned endpoints; `attribution` codes
   match a hand-checked high-AUC fixture.
8. **Property-based (hypothesis)**: random score distributions ⇒ band within
  [0,1], monotone, contains the empirical curve; scores under strictly
   monotone transforms ⇒ identical envelope band (rank invariance), while WH
   band changes — the paper's key contrast, as a test.
9. **Plot smoke tests**: figures build headless (`Agg`), legend labels
  present, no matplotlib ⇒ actionable `RocciError` (simulated via
   import-block fixture).

Coverage goal: ≥ 95% for `python/rocci`, enforced in CI.

---

## 12. Developer tooling, guidelines, and infrastructure

### 12.1 Toolchain

- **Python env**: `uv` throughout (`uv sync` installs the dev group and builds
the extension in-place via the maturin build backend; `uv run maturin develop --release` for an optimized local build when benchmarking). `uv.lock` is
committed for reproducible dev/CI environments; runtime pins in
`pyproject.toml` stay loose (`numpy>=1.24`, `scipy>=1.10`) because rocci is a
library.
- **Rust**: pinned via `rust-toolchain.toml` (a specific stable, e.g.
`1.96`, plus `rustfmt` and `clippy` components) so every contributor and CI
runner compiles with the same compiler. MSRV = the pinned version − 2 minors,
checked in CI, documented in Cargo.toml `rust-version`.
- **Single version source**: `version` lives in `pyproject.toml`
(`[project]`); Cargo.toml carries an independent internal crate version that
never ships. `rocci.__version__` reads `importlib.metadata.version`.

### 12.2 Canonical commands (`justfile`)

One entry point so "how do I run X" has exactly one answer, identical locally
and in CI (CI calls the same recipes):

```
just setup          # uv sync + maturin develop + pre-commit install
just test           # pytest -x -q (fast suite; excludes slow/calibration)
just test-all       # full matrix incl. slow + ROCCI_BACKEND=numpy pass
just lint           # ruff check + ruff format --check + cargo fmt --check + cargo clippy -D warnings
just fix            # ruff check --fix + ruff format + ty check --fix + cargo fmt
just typecheck      # ty check
just rust-test      # cargo test
just bench          # benchmarks/ perf gates, prints table vs thresholds
just docs           # mkdocs serve with executed vignettes
just fixtures       # regenerate golden masters (requires studroc_paper checkout; prints provenance)
just release-prep X.Y.Z   # version bump + git-cliff changelog + absolute perf check (§14.5)
```

### 12.3 Pre-commit hooks (`.pre-commit-config.yaml`)

Fast, format-level only — heavy checks belong to CI, not the commit path:
`ruff check --fix`, `ruff format`, `cargo fmt`, end-of-file/trailing-whitespace,
`check-added-large-files` (blocks accidental fixture bloat; golden masters are
added deliberately via `git add -f` paths documented in CONTRIBUTING). ty and
clippy run in CI and in `just lint`/`just typecheck`, not pre-commit.

### 12.4 Code guidelines (enforced, not aspirational)

**Lint/format: ruff** (checker *and* formatter — no black, no standalone
isort). Ruleset:

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = [
    "E", "W",    # pycodestyle
    "F",         # pyflakes
    "I",         # isort-style import sorting
    "B",         # bugbear (likely bugs)
    "UP",        # pyupgrade (keep idioms current for the 3.10 floor)
    "D",         # pydocstyle — Google convention
    "N",         # pep8-naming
    "NPY",       # numpy-specific rules (legacy RNG ban, deprecated aliases)
    "C4",        # comprehension hygiene
    "SIM",       # simplifications
    "RET",       # return-statement hygiene
    "ARG",       # unused arguments
    "PT",        # pytest style
    "TID252",    # ban relative parent imports
    "PERF",      # perflint (accidental quadratic patterns)
    "RUF",       # ruff-native rules (incl. mutable-default, unused-noqa)
]
ignore = [
    "D105",      # magic-method docstrings
    "D107",      # __init__ docstrings (class docstring carries the contract)
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.isort]
known-first-party = ["rocci"]
split-on-trailing-comma = false

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D", "ARG"]        # test names are the documentation
"benchmarks/**" = ["D"]
"docs/**" = ["D", "E402"]        # vignettes import mid-file for narrative flow

[tool.ruff.format]
quote-style = "double"
docstring-code-format = true
skip-magic-trailing-comma = true
```

**Type checking: ty** (Astral's checker — not mypy). Fast enough to run on
every PR without caching gymnastics, and `ty check --fix` applies its
autofixes (e.g. stale suppression comments), wired into `just fix`. Config:

```toml
[tool.ty.environment]
python-version = "3.10"          # check against the floor, not the dev interpreter

[tool.ty.src]
include = ["python/rocci", "tests"]

[tool.ty.rules]
# ty's default correctness rules stay at their default (error) severity.
# Promote the off-by-default/lenient ones that catch real bugs in numeric code:
possibly-unresolved-reference = "error"
unused-ignore-comment = "error"   # suppressions must stay honest
redundant-cast = "warn"
division-by-zero = "warn"
```

Suppressions use `# ty: ignore[rule]` with the rule name spelled out, never
bare; `unused-ignore-comment = "error"` plus autofix keeps them from
accumulating. Tests are checked but, as a gradual checker, ty does not demand
annotations there — annotate test helpers, not every test body.

- Google docstrings on every public object; every public docstring includes a
runnable `Examples:` block — doctests are executed in CI (§14.1), so examples
cannot rot.
- Rust: `cargo clippy --all-targets -- -D warnings` with `pedantic` enabled and
a short, documented allow-list; `unsafe` only at the FFI boundary, each block
with a `// SAFETY:` comment (clippy `undocumented_unsafe_blocks` = deny).
- Comments/docstrings are evergreen; no commented-out code; no TODOs without a
linked issue number.

### 12.5 Contribution & repo conventions

- **CONTRIBUTING.md** covers: dev setup on all three OSes (rustup + uv +
`just setup`; explicit Windows note that MSVC Build Tools are required),
the backend override (`ROCCI_BACKEND=numpy`) for debugging, how to run the
slow statistical suite locally, golden-master regeneration policy (§5.7
fixtures change **only** with a spec change and a PR explaining the
statistical delta), and the PR checklist.
- **Branch model**: trunk-based; `main` protected (required checks = **every**
CI job: lint, typecheck, audit, rust, test matrix, fallback, minimums,
docs build, **calibration**, **perf** — the heavy gates run per-PR and are
required to merge, §14.2; linear history; no direct pushes). Short-lived
feature branches; squash merge.
- **Commits/changelog**: Conventional Commits enforced by a CI title check on
PRs (squash merge makes PR title = commit); `git-cliff` generates
CHANGELOG.md at release time (§14.5).
- **Issue templates**: bug template requires `rocci.show_versions()` output
and a minimal reproducer; feature template asks "does this belong in the
validated statistical surface?" to guard the deliberately small API.
- **CODEOWNERS**: owner on everything; `python/rocci/band/` and `rust/`
additionally flagged as "statistical core — golden masters must pass, spec
update required" in the PR template.
- **Dependabot**: weekly, ecosystems = github-actions, cargo, pip (dev group);
security audits in CI via `cargo audit` and `pip-audit` (§14.4).

---

## 13. Human-facing documentation

### 13.1 Stack

`mkdocs-material` + `mkdocstrings[python]` for API reference from docstrings +
`mkdocs-jupyter` for executed vignettes + `mike` for versioned deployment to
GitHub Pages (`latest` alias tracks main, `stable` tracks the latest release;
version picker in the header). Math via KaTeX (`pymdownx.arithmatex`).

### 13.2 Site map

```
Home                      # hero figure + 5-line quickstart + "why rocci"
Getting started
├── Installation          # pip / conda-forge / fallback-backend note
└── Quickstart            # the 5-line example, annotated output figure
User guide
├── Which band should I use?    # condensed paper §1–3: WH failure modes, KS vacuity, envelope;
│                               # incl. the multiclass asymmetry (envelope composes via OvR+Bonferroni,
│                               # WH anti-composes: rest-class mixtures are its failure mode)
├── Using rocci with your data  # ingestion guide: sklearn, torch logits, statsmodels, PyMC/arviz, pandas/polars
├── Reading the band            # band semantics: simultaneous vs pointwise, vacuous region, floors
├── Diagnostics                 # floor attribution plots, normality report, worked "bad WH" example
└── Performance                 # backend model, n_threads, expected timings table (§9)
Method
├── The envelope method         # written from spec §5 + appendix A1–A10; paper figures as illustrations
└── Theoretical behavior        # condensed from the paper's theory report (content source only)
Vignettes                       # §13.3, executed notebooks
API reference                   # mkdocstrings: roc_band, roc_band_ovr, from_estimator, RocBand, warnings
FAQ                             # logits vs probabilities; why lower band is 0 at tiny FPR;
                                # ties/discrete scores (safe: conservative, tested); multiclass (why roc_band
                                # fails, what roc_band_ovr guarantees)
Changelog
Contributing
```

### 13.3 Vignettes

Stored as jupytext Markdown (`docs/vignettes/*.md`) — clean diffs, no committed
outputs — and **executed during the docs build** so they can never silently
rot; a docs-build failure is a real regression signal. Four vignettes:

1. **Quickstart with scikit-learn** — breast-cancer dataset, logistic
  regression, `from_estimator`, `band.plot()`, reading `band.summary()`.
2. **Deep-learning scores** — torch-produced logits with heavy tails; shows
  `normal=True` firing `NormalityWarning`, WH band visibly wrong vs the
   envelope; demonstrates the rank-invariance point (logits vs sigmoid
   probabilities give identical envelope bands). Uses a pre-saved score array
   so the docs build does not depend on torch.
3. **Bayesian workflow** — PyMC-style posterior predictive probabilities
  (pre-saved draws), `score_reduce="mean"`, and an honest discussion of what
   the band does and does not capture.
4. **Anatomy of the band** — `diagnostics=True`, floor attribution across
  n ∈ {50, 500, 5000} at high AUC; teaches the yellow/green/interior regions
   and the vacuous-region boundary.

### 13.4 README (repo + PyPI landing page)

Hero figure (envelope vs WH on heavy-tailed data — the paper's money shot),
badge row, the 5-line quickstart, one paragraph of "when to use this", link to
docs, citation block. The README is the elevator pitch; everything longer
lives in the docs site.

Badge row (top of README, in this order; all resolve automatically once the
services exist — no manual upkeep):


| Badge               | Source                                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------- |
| PyPI version        | `img.shields.io/pypi/v/rocci` → pypi.org/project/rocci                                       |
| Python versions     | `img.shields.io/pypi/pyversions/rocci` (driven by classifiers)                               |
| conda-forge version | `img.shields.io/conda/vn/conda-forge/rocci` (appears once the feedstock lands)               |
| CI                  | `github.com/ndelaneybusch/rocci/actions/workflows/ci.yml/badge.svg`                          |
| Merge gates         | `.../workflows/gates.yml/badge.svg` (calibration + perf — the "the science is tested" badge) |
| Coverage            | Codecov badge for the repo                                                                   |
| Docs                | static shields.io "docs — latest" badge linking to the Pages site                            |
| License             | `img.shields.io/pypi/l/rocci`                                                                |
| DOI                 | Zenodo concept-DOI badge (appears after the first release, §13.5)                            |


### 13.5 Citation & archival

`CITATION.cff` (package) referencing the method paper/preprint once it has a
DOI; Zenodo webhook mints a DOI per GitHub release. `band.summary()` ends with
a one-line "please cite" pointer.

---

## 14. CI and release engineering

### 14.1 `ci.yml` — every PR and push to main

Concurrency group per-ref (cancel superseded runs). Caching: `Swatinem/rust-cache`
for cargo, uv's built-in cache keyed on `uv.lock`. Jobs:


| Job             | Matrix                                     | Content                                                                                                                                                    |
| --------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lint`          | ubuntu                                     | `just lint` (ruff check+format, cargo fmt, clippy -D warnings) + Conventional-Commit PR title check                                                        |
| `typecheck`     | ubuntu                                     | `ty check` (§12.4 config)                                                                                                                                  |
| `audit`         | ubuntu                                     | `cargo audit` + `pip-audit` (fast; runs per-PR, not on a schedule)                                                                                         |
| `rust`          | ubuntu, windows                            | `cargo test` (oracle + determinism kernel tests), MSRV build check                                                                                         |
| `test`          | {ubuntu, windows, macos-14} × {3.10, 3.13} | maturin build + full pytest (minus `slow`), doctests (`--doctest-modules` on `python/rocci`), coverage upload (Codecov, ≥95% gate on the ubuntu/3.13 cell) |
| `test-fallback` | ubuntu × 3.10                              | `ROCCI_BACKEND=numpy` full pytest — proves the package works with zero compiled code                                                                       |
| `test-minimums` | ubuntu × 3.10                              | pins `numpy==1.24.*`, `scipy==1.10.*` — guards the loose lower bounds                                                                                      |
| `docs-build`    | ubuntu                                     | `mkdocs build --strict` with vignette execution — broken docs/vignettes fail PRs                                                                           |


Python 3.10 and 3.13 only on the matrix — abi3 wheels make the intermediate
versions mechanically identical, so a wider sweep buys nothing per-PR.

`docs.yml` deploys the site: on push to main, `mike deploy latest`; release
tags additionally run `mike deploy X.Y stable` (§14.5). PR runs only build
(`docs-build` above) — deployment never happens from a PR.

### 14.2 `gates.yml` — the heavy merge gates, per-PR and required

No scheduled/nightly runs anywhere: every check that protects main runs on the
PR itself and is a **required status check** in branch protection — nothing
lands without it having passed on the merge candidate. Two jobs, both designed
to be *deterministic* (a required check that flakes trains people to override
it):

- `**calibration`** — the statistical suite (§11.5) with **fixed seeds**, so
it is a deterministic regression test of the assembled method, not a
stochastic gate: four DGPs (A15, including the ties cell) × n ∈ {30, 200} × 250 sims, asserting coverage
within precomputed Monte-Carlo bands around nominal (near nominal from
above — closer is better, not higher) and mean width below the KS band.
Sized to finish in ≤ 15 min on a standard ubuntu runner; parallelized with
`pytest-xdist`.
- `**perf`** — the §9 gates, made runner-noise-proof by measuring **relative
to main**: the job builds both the PR head and the merge-base from main in
the same runner, runs the benchmark suite on each, and fails on a > 30%
regression in any kernel timing or any absolute-budget breach at 2× the §9
thresholds. Absolute §9 numbers remain the release gate, verified on
developer hardware via `just bench` before tagging.

Both jobs are in the required-checks list alongside every `ci.yml` job
(§12.5 branch protection). `concurrency` cancels superseded runs so PR
iteration stays cheap.

### 14.3 `wheels.yml` — release candidates and releases

Trigger: version tags (`v`*) and manual dispatch (dry-run mode that builds but
skips publish). Built with `PyO3/maturin-action`, abi3-py310 (one wheel per
platform covers every Python ≥3.10):


| Target                | Runner                             |
| --------------------- | ---------------------------------- |
| manylinux2014 x86_64  | ubuntu                             |
| manylinux2014 aarch64 | ubuntu-24.04-arm (native, no QEMU) |
| musllinux_1_2 x86_64  | ubuntu                             |
| macOS x86_64          | macos-13                           |
| macOS arm64           | macos-14                           |
| Windows x64           | windows                            |
| sdist                 | ubuntu                             |


Every wheel is **smoke-tested in a clean venv on its own platform before
publish**: `pip install <wheel>` + import + a 500-sample `roc_band` end-to-end

- assert `backend == "rust"`. The sdist is smoke-tested by compiling from
source on ubuntu (validates that source installs work where no wheel matches).
`pip install rocci` must never compile Rust on a supported platform — a CI
assertion, not a hope.

### 14.4 Publishing & supply chain

- **PyPI trusted publishing** (OIDC) — no long-lived tokens; publish job gated
on all wheel smoke tests, isolated in a GitHub `release` environment.
- GitHub artifact attestations (`actions/attest-build-provenance`) on wheels.
- Release candidates (`vX.Y.ZrcN` tags) publish to **TestPyPI** through the
same pipeline; final tags go to PyPI.

### 14.5 Tag + release flow (documented in CONTRIBUTING, automated end-to-end)

The tag is the single trigger; everything downstream is machine-checked so a
release cannot ship half-done or inconsistent.

1. **Release PR**: `just release-prep X.Y.Z` bumps `pyproject.toml`, runs
  `git-cliff` to regenerate CHANGELOG.md, and runs `just bench` to verify the
   §9 *absolute* budgets on developer hardware (the CI perf gate is relative,
   §14.2). PR titled `release: vX.Y.Z`; passes every required gate like any
   other PR.
2. **Tag**: after merge, an annotated tag `vX.Y.Z` is pushed on the merge
  commit. Tags are the only publish trigger; `workflow_dispatch` offers a
   dry-run that builds and smoke-tests but cannot publish.
3. `**release-guard` job** (first in `wheels.yml`, everything depends on it):
  asserts tag name == `pyproject.toml` version == top CHANGELOG section, tag
   commit is an ancestor of `main`, and the tag is annotated. Any mismatch
   aborts before a single wheel is built.
4. **Build + verify**: the §14.3 wheel matrix builds; each wheel is
  smoke-tested in a clean venv on its own platform (install → import →
   500-sample `roc_band` → `backend == "rust"`); sdist compiles from source.
5. **Publish**: gated on all smoke tests, runs in the GitHub `release`
  environment (trusted publishing/OIDC; environment protection rule =
   manual approval by the owner — the human "go" lives here, not in a
   fragile earlier step). `vX.Y.ZrcN` tags publish to TestPyPI; final tags to
   PyPI. Provenance attestations attached.
6. **Post-publish verification**: a final job polls PyPI until the version is
  live, then `pip install rocci==X.Y.Z` in clean environments on all three
   OSes and re-runs the smoke test — catching broken metadata/wheel-selection
   issues the moment they exist, not when the first user does.
7. **Announce artifacts**: GitHub Release created with the changelog section
  (this triggers Zenodo's DOI webhook); `docs.yml` runs
   `mike deploy X.Y stable`.
8. **conda-forge**: after the first PyPI release, submit the feedstock via
  staged-recipes (standard Rust/maturin pattern: `{{ compiler('rust') }}` +
   `{{ stdlib('c') }}` in build, `maturin` + `pip` in host; build from the
   PyPI sdist). Subsequent releases arrive as automatic regro-cf-autotick-bot
   PRs — merging them is the only recurring conda maintenance. Acceptance for
   "conda-ready": `conda install -c conda-forge rocci` works on
   linux-64/aarch64, osx-64/arm64, win-64.

---

## 15. Build order (milestones)

1. **M0 — Skeleton**: repo, maturin build, CI green on hello-world kernel,
  wheel matrix proven (this is the riskiest infrastructure; front-load it). DONE.
2. **M1 — Statistical core**: grids + empirical ROC, floors, envelope assembly
  in NumPy against golden masters (§5.7) using a temporary NumPy kernel. DONE.
3. **M2 — Rust kernel**: port the profiled prototype into PyO3, oracle +
  determinism tests, fallback parity, backend selection. DONE
4. **M3 — API + ingestion**: `roc_band`, `RocBand`, `from_estimator`,
  warnings, ingestion matrix tests. DONE.
5. **M4 — WH path + diagnostics**: §6 + normality machinery. DONE.
6. **M5 — Plots + docs + vignettes**: §7, mkdocs site (§13), executed
  vignettes, versioned deployment.
7. **M6 — Merge gates**: calibration suite (§11.5) and perf gates (§9) wired
  into `gates.yml` as required PR checks (§14.2).
8. **M7 — Release**: 0.1.0 to PyPI, then conda-forge feedstock.

Definition of done for v0.1.0: all gates in §9 and §11 green; a user with only
`pip` gets a correct, fast band and a paper-quality figure in ≤ 5 lines on
Windows/macOS/Linux without compiling anything.