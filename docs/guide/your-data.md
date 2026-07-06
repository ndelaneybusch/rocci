# Using rocci with your data

Ingestion is duck-typed: rocci accepts what the ecosystem produces without
importing any of the producing libraries. That means it works with all the usual
scientific computing libraries, but also with custom types - just include an
`__array__` or `.to_numpy()` method. Anything array-like works; anything
ambiguous raises a `RocciError` that names the fix.

## Containers

| You have | It works because |
|---|---|
| NumPy arrays, Python lists/tuples | used directly |
| pandas `Series` / `Index` | `__array__` protocol |
| polars `Series` | `.to_numpy()` duck-call |
| torch / JAX / CuPy tensors | `__dlpack__`; CUDA tensors fall back to `.detach().cpu().numpy()` |
| xarray / arviz `DataArray` | `__array__` protocol |

Column vectors — `y.reshape(-1, 1)` labels or `predict_proba(X)[:, [1]]`
scores — are recognized and raveled.

## Labels (`y_true`)

Booleans, `{0, 1}`, and `{-1, 1}` infer the positive class automatically.
Any other pair of values (strings included) works with an explicit
`pos_label=`:

```python
roc_band(diagnosis, score, pos_label="malignant")
```

More than two distinct labels is an error by design — there is no single ROC
curve for a multiclass problem. Use
[`roc_band_ovr`][rocci.roc_band_ovr] for one-vs-rest bands with a family-wise
guarantee, or binarize and pass `pos_label`.

## Scores (`y_score`)

- **1-D scores** — used directly. Higher must mean "more positive".
- **`(n, 2)` probability matrices** (rows summing to 1, e.g.
  `predict_proba` output) — the positive-class column is selected
  automatically, recorded as an INFO note in `band.summary()`.
- **Posterior draws** — `(draws, n)` or `(chain, draw, n)` arrays (PyMC,
  arviz) require `score_reduce="mean"` or `"median"`; rocci reduces over the
  draw axes and notes that the band then quantifies sampling uncertainty of
  the reduced scores, not posterior uncertainty. The
  [Bayesian vignette](../vignettes/03-bayesian-workflow.md) discusses what
  that means in practice.
- **Multiclass `predict_proba`** (`(n, m>2)`, rows summing to 1) — an error
  pointing you to `roc_band_ovr`.

### Logits or probabilities?

Either. The envelope band is invariant to strictly monotone transforms of the
scores, so logits and their sigmoid give the identical band — no calibration
step needed first. (The Working–Hotelling band does *not* have this property;
see [Which band should I use?](which-band.md).)

### Ties, discreteness, ±inf, NaN

- **Ties and discrete scores never fail.** The band stays valid and errs
  conservative; heavy ties (fewer than half the pooled scores distinct) get a
  `TiesWarning` saying exactly that. Even a class with entirely constant
  scores proceeds — the band degenerates to the exact floors and is honestly
  wide.
- **±inf scores are legal** (they sort; a strictly monotone squashing would
  give the same band anyway).
- **NaN raises by default.** Pass `nan_policy="omit"` to drop NaN rows (in
  either input) with a warning stating the count.

## From estimators

```python
from rocci import from_estimator
band = from_estimator(clf, X_test, y_test)
```

Uses `predict_proba` when the estimator has it, else `decision_function`
(override with `response_method=`). Works with scikit-learn pipelines,
statsmodels results wrapped in a thin shim, or anything else exposing either
method.

## Small samples

Classes with fewer than 2 samples are an error; fewer than 20 gets a
`SmallSampleWarning`. The band remains valid at any size — the exact floors
take over and the band gets honestly wide instead of quietly overconfident.
