# How rocci is verified

A confidence band is only as trustworthy as the code that computes it. The
statistical case for the method lives in [Theoretical behavior](theory.md); this
page is the *engineering* case — the evidence that the implementation faithfully
realizes that method and keeps doing so.

The short version: **no layer is trusted on its own.** The band you get back has
been checked from four independent directions — against the validated reference
implementation, against hand-derived oracles component by component, against
known ground truth as a whole, and against itself under adversarial and
cross-implementation stress — and every one of those checks runs in CI on every
change. What follows is the case in totality, and where to look for each part.

## It matches the validated reference, exactly

The strongest bar rocci clears is not a property test but an *identity* test.
The reference implementation used to validate the method in the paper was run
once to record golden-master fixtures; given the exact inputs it recorded,
rocci's assembly must reproduce its band to `atol=1e-6`. The precedence rule is
absolute: if the code and a fixture ever disagree, **the fixture wins** —
fixtures are never regenerated to make new code pass. This turns "we reproduce
the validated numbers" from a claim into a gate.

*See:* `tests/test_golden_master.py`, and the recording provenance in
`tests/fixtures/golden/PROVENANCE.md`.

## Every component is checked against an independent oracle

Self-consistent code can be consistently wrong. So each primitive is compared
not to itself but to a definition derived independently:

- the fast empirical ROC matches a naive $O(n^2)$ threshold-and-count reference
  bit-for-bit, including under ties and $\pm\infty$;
- the NumPy bootstrap kernel is checked **bit-for-bit** against a brute-force
  oracle that expands the resamples and counts by hand off the *same* RNG
  stream — the one test that can see an off-by-one or tie bug the statistics
  would hide;
- `auc` matches a literal pairwise Mann–Whitney count (ties weighted ½);
- the Wilson bounds are held to the score-test inversion they solve,
  $(\hat p - p)^2 = z^2 p(1-p)/n$, not to a library round-trip;
- the Beta floor is checked through the Beta/Binomial survival identity, which
  catches swapped distribution parameters that a quantile round-trip cannot;
- the Working–Hotelling band reproduces an independent transcription of its
  closed forms to float precision.

*See:* `tests/test_grids.py`, `tests/test_fallback_kernel.py`,
`tests/test_envelope.py`, `tests/test_floors.py`, `tests/test_normal.py`.

## The assembled band is calibrated against ground truth

Components that each match their formula can still assemble into a band that
misses the true curve. The end-to-end calibration gate rules this out: it draws
from data-generating processes with *closed-form* population ROCs chosen to be
unkind — binormal, heavy-tailed $t_3$, bimodal-negative mixtures, and
discretized heavy-tie scores — at sample sizes as small as 30 per class, and
requires two things at once. Coverage must sit at or above nominal (an
undercovering band fails), **and** the mean width must be strictly below the
proven KS/DKW reference band — so a band cannot pass by being trivially wide.

*See:* `tests/test_calibration.py` (the full multi-DGP gate, marked `slow`) and
`tests/test_statistics.py` (its fast single-draw companions: true-ROC
containment, the KS-width win, and correct response to confidence and $n$).

## Invariants hold on inputs nobody chose

Curated examples only prove behavior on inputs someone thought of.
Property-based tests attack the same invariants with random sizes, shifts, and
four distribution families, and confirm the band is always well-formed (ordered,
monotone, pinned, vacuous-below-$q_1$) and exactly rank-invariant under
monotone rescaling. The kernel gets the same treatment with a fuzzer that
includes $\pm\infty$ and massive tie pressure.

*See:* `tests/test_properties.py`, `tests/test_fallback_kernel.py`.

## Two independent implementations agree

rocci ships two kernels — compiled Rust and vectorized NumPy — written
separately, with different RNG streams. A CI contract holds them to
distributional agreement at 8 000 replicates, so a bug would have to exist
*identically* in both to go unnoticed. Each kernel is independently reproducible
bit-for-bit across thread counts and repeated runs.

*See:* `tests/test_rust_backend.py`, `tests/test_backend.py`.

## It is reproducible and robust at the boundary

The reproducibility contract — same seed, backend, and version yields a
bit-identical band regardless of thread count — is itself tested, not just
documented. The package boundary is guarded too: ingestion is red-teamed across
every container and label/score edge case it accepts, and a clean-subprocess
test enforces that the runtime dependencies really are numpy + scipy only, with
every optional-feature path failing as an actionable `RocciError` rather than a
raw import traceback.

*See:* `tests/test_api.py`, `tests/test_ingest.py`, `tests/test_optional_deps.py`.

## Performance is gated, not hoped for

Speed is held to absolute budgets, enforced by a benchmark job in CI (2 000
replicates on 100 000 samples in well under half a second; see
[Performance](../guide/performance.md) for the table). Memory stays bounded, and
because the RNG is keyed per replicate rather than per thread, the numbers do not
depend on scheduling.

*See:* `benchmarks/`, and the `just bench` recipe.

## It stays verified

None of the above is a one-time audit. CI runs the same `just` recipes a
developer runs locally — the fast suite, the full cross-backend and `slow`
statistical matrix, the Rust tests, doctests on every public example, `mkdocs
build --strict` with the vignettes *executed* (so documentation cannot silently
rot), plus lint, format, and type checks. You can reproduce the whole thing with
`just test-all`.

*See:* the `justfile` and [Contributing](../contributing.md).

## The honest edges are part of the case

Confidence is strengthened, not weakened, by naming where the guarantee stops.
The envelope's finite-sample simultaneous coverage is *calibrated* rather than
proven — which is exactly why the calibration gate and the proven KS/DKW
yardstick exist. The vacuous region at tiny FPR is a theorem, not a
shortcoming: no distribution-free lower bound exists there, and rocci reports the
boundary instead of drawing an unjustifiable curve. The parametric
`normal=True` band degrades under misspecification, so it is diagnosed and
warned rather than certified. These edges are characterized and surfaced to the
user; [Theoretical behavior](theory.md) is where they are spelled out.
