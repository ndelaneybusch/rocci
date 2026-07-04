# Quickstart

The five-line version:

```python
from rocci import roc_band

band = roc_band(y_true, y_score, random_state=0)
band.plot()
print(band.summary())
```

`y_true` is anything label-like (bools, `{0, 1}`, `{-1, 1}`, or any two
values plus `pos_label=`); `y_score` is anything score-like (higher = more
positive). Arrays, Series, tensors, and lists all work —
[Using rocci with your data](../guide/your-data.md) has the full matrix.

## What you get back

`roc_band` returns a frozen [`RocBand`](../api.md) carrying the band and its
provenance:

```python
band.fpr, band.tpr          # the empirical ROC on the FPR grid
band.lower, band.upper      # the simultaneous band arms
band.auc, band.auc_ci       # Mann-Whitney AUC (= sklearn) + bootstrap CI
band.confidence             # the simultaneous coverage target you asked for
band.at([0.05, 0.10])       # (lower, tpr, upper) at any FPR you care about
band.to_dataframe()         # pandas, if you have it
```

A typical `summary()`:

```text
rocci confidence band (envelope)
  samples: n_neg=212, n_pos=357
  coverage: 95% simultaneous
  AUC: 0.9974  (CI: 0.9926, 1.0000)
  band area (mean width): 0.0534
  backend: rust, n_boot=2000
  no distribution-free lower bound exists below FPR ~= 0.0125; increase the
  number of negatives to certify lower FPRs.
  please cite rocci (see CITATION.cff).
```

Two lines deserve a first-time explanation:

- **"95% simultaneous"** means the *whole* true curve stays inside the band
  in 95% of datasets — strictly stronger than 95% pointwise intervals, which
  are individually right but jointly almost surely wrong somewhere.
  [Reading the band](../guide/reading-the-band.md) unpacks this.
- **The vacuous-region line** is rocci being honest: below an FPR of about
  `1/n_neg`-ish, *no method* can certify a distribution-free lower bound, so
  the lower band is 0 there rather than pretending otherwise.

## From a fitted classifier

```python
from rocci import from_estimator

band = from_estimator(clf, X_test, y_test, random_state=0)
```

Duck-typed like scikit-learn's `RocCurveDisplay.from_estimator`: uses
`predict_proba` when available, else `decision_function` — but works with any
object exposing either method, no sklearn import required.

## Reproducibility

Pass `random_state=` to seed the bootstrap. Same seed + same backend + same
rocci version ⇒ a bit-identical band, independent of thread count.

Run the [scikit-learn vignette](../vignettes/01-quickstart-sklearn.md) to see
all of this executed on real data.
