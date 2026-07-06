# How rocci is verified

[Simulations and validation](simulations.md) and
[Theoretical behavior](theory.md) make the statistical case for the method.
This page makes the engineering case that the implementation computes that
method correctly. Four checks carry the weight:

1. **Reference identity** — the assembly reproduces the validated reference
   implementation on recorded inputs.
2. **Component oracles** — every statistical primitive matches an
   independently derived definition.
3. **End-to-end calibration** — the assembled band covers known population
   ROCs at nominal confidence while staying narrower than a band with proven
   coverage.
4. **Independent implementations** — two separately written kernels must
   agree.

All of it runs in CI on every change.

## 1. Reference identity

The reference implementation from the
[method validation study](https://github.com/ndelaneybusch/studroc_paper) was
run once to record golden-master fixtures: exact inputs and the bands they
produce. When code and fixture disagree, the fixture wins — fixtures are
never regenerated to make new code pass, and change only through a PR that
justifies the statistical delta and its coverage consequences.

**The bar:** on each fixture's recorded inputs, rocci's band equals the
validated reference band to `atol=1e-6`.

*Tests:* `tests/test_golden_master.py`; recording provenance in
`tests/fixtures/golden/PROVENANCE.md`.

## 2. Component oracles

Self-consistent code can be consistently wrong, so each primitive is compared
to a definition derived independently of the implementation:

- the empirical ROC equals a naive $O(n^2)$ threshold-and-count reference,
  including under ties and $\pm\infty$;
- the NumPy bootstrap kernel equals a brute-force oracle that expands the
  resamples and counts by hand from the same RNG stream — the one test that
  can catch an off-by-one or tie-handling bug the statistics would average
  away;
- `auc` equals the literal pairwise Mann–Whitney count (ties weighted ½);
- the Wilson bounds satisfy the score-test equation they invert,
  $(\hat p - p)^2 = z^2 p(1-p)/n$, rather than round-tripping through a
  library;
- the Beta floor satisfies the Beta/Binomial survival identity, which catches
  swapped distribution parameters that a quantile round-trip cannot;
- the Working–Hotelling band matches an independent transcription of its
  closed forms.

**The bar:** bit-for-bit equality where the definition is discrete (ROC,
kernel, AUC); float precision where it is a closed form.

*Tests:* `tests/test_grids.py`, `tests/test_fallback_kernel.py`,
`tests/test_envelope.py`, `tests/test_floors.py`, `tests/test_normal.py`.

## 3. End-to-end calibration

Components that individually match their formulas can still assemble into a
band that misses the true curve. The calibration gate draws from
data-generating processes with closed-form population ROCs — binormal,
heavy-tailed $t_3$, bimodal-negative mixtures, and discretized heavy-tie
scores — at sample sizes down to 30 per class.

**The bar:** on every process, jointly, (a) coverage at or above nominal —
an undercovering band fails — and (b) mean width strictly below the KS/DKW
reference band, whose coverage is proven — so a band cannot pass by being
wide.

*Tests:* `tests/test_calibration.py` (the full multi-DGP gate, marked
`slow`); `tests/test_statistics.py` (fast single-draw companions: true-ROC
containment, the KS-width win, correct response to confidence and $n$).

## 4. Independent implementations

The Rust and NumPy kernels were written separately and use different RNG
streams, so a bug must exist identically in both to pass their contract test.

**The bar:** distributional agreement between the kernels at 8 000
replicates, and bit-identical output within each kernel for a fixed seed,
regardless of thread count or repeated runs.

*Tests:* `tests/test_rust_backend.py`, `tests/test_backend.py`.

## Supporting checks

- **Invariants on random inputs.** Property-based tests generate random
  sizes, shifts, and four distribution families, and assert the band is
  always well-formed (ordered, monotone, pinned, vacuous below $q_1$) and
  exactly rank-invariant under monotone rescaling. A kernel fuzzer includes
  $\pm\infty$ and heavy tie pressure.
  (`tests/test_properties.py`, `tests/test_fallback_kernel.py`)
- **The package boundary.** Ingestion is exercised across every container
  and label/score edge case it accepts; a clean-subprocess test confirms the
  runtime dependencies really are numpy and scipy only, with every
  optional-feature path failing as an actionable `RocciError` rather than a
  raw import traceback.
  (`tests/test_api.py`, `tests/test_ingest.py`, `tests/test_optional_deps.py`)
- **Reproducibility.** The contract — same seed, backend, and version gives
  a bit-identical band — is itself tested, not just documented.
  (`tests/test_api.py`)
- **Performance budgets.** Absolute wall-clock and memory budgets are
  enforced by a benchmark job in CI; see
  [Performance](../guide/performance.md) for the numbers.
  (`benchmarks/`, `just bench`)

## Enforcement

None of this is a one-time audit. CI runs the same `just` recipes a
developer runs locally, on every change: the fast suite, the full
cross-backend and `slow` statistical matrix, the Rust tests, doctests on
every public example, and `mkdocs build --strict` with the vignettes
executed, plus lint, format, and type checks. `just test-all` reproduces the
full matrix locally.

*See:* the `justfile` and [Contributing](../contributing.md).

## What is not claimed

- The envelope's finite-sample simultaneous coverage is calibrated, not
  proven. That is why the calibration gate exists and why the proven KS/DKW
  band is its yardstick.
- No distribution-free lower bound exists at FPR below roughly
  $1/n_\text{neg}$. rocci reports the boundary (`band.vacuous_below`) rather
  than drawing a curve it cannot justify.
- The `normal=True` band degrades under misspecification. It is diagnosed
  and warned, not certified.

[Theoretical behavior](theory.md) spells each of these out.
