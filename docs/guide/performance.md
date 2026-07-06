# Performance

The bootstrap kernel — the only computationally heavy part of a band — is
compiled Rust, parallelized with replicate-indexed RNG streams so results are
bit-identical regardless of thread count.

## Expected timings

End-to-end `roc_band` at the defaults (`n_boot=2000`, `K=512`), desktop-class
12-core hardware:

| total samples | end-to-end budget |
|---|---|
| 10 000 | < 100 ms |
| 100 000 | < 500 ms |
| 1 000 000 | < 3 s |

These are the release gates, enforced by a benchmark job in CI; current
hardware typically runs well under them. Kernel cost scales as
$O(B \cdot (n + K))$; everything outside the kernel (ingestion, empirical
ROC, floors, assembly) is $O(n \log n)$ and budgeted under 50 ms at
n = 100 000.

Memory: no allocation proportional to $B \times n$ — peak extra memory is the
$B \times K$ output matrix (8 bytes/entry; ~8 MB at the defaults) plus
$O(n)$ per thread.

## `n_threads`

`None` (default) uses all cores; `-1` also means all cores (matching
sklearn's `n_jobs` convention); any positive integer pins the pool size.
Thread count never changes results, only wall-clock time:

```python
roc_band(y, s, n_threads=4)
```

## `n_boot` and `grid_size`

- `n_boot=2000` (default) is a good general-purpose setting. Below 1000 the
  envelope's quantile resolution gets coarse (rocci warns); below 100 it
  refuses. Raising to 5000–10000 sharpens the band's boundary slightly at
  proportional cost.
- `grid_size=None` uses `min(512, n_neg + 1)` FPR grid points — one per
  achievable FPR step up to 512. There is rarely a reason to override it.

## The two backends

| backend | what | when |
|---|---|---|
| `rust` | compiled kernel, rayon-parallel | any platform with a wheel (all mainstream ones) |
| `numpy` | vectorized fallback, identical semantics | no wheel and no toolchain; `FallbackBackendWarning` once per process |

The fallback is 10–30× slower than Rust — still far faster than naive
sort-based bootstrapping, and adequate for interactive use up to n ≈ 10⁵.
If a run is unexpectedly slow, check `band.backend` or
`rocci.show_versions()`.

## Reproducibility contract

`random_state=seed` ⇒ bit-identical band for the same backend and rocci
version, independent of `n_threads` and scheduling. Across backends (rust vs
numpy), bands agree statistically, not bit-wise — the two kernels use
different RNG streams by design, and a CI contract test holds them to
distributional agreement.
