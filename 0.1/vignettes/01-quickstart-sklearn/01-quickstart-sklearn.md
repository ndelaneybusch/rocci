---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Quickstart with scikit-learn

The full workflow on real data: fit a classifier, hand it to rocci, read the
band. We use the diabetes dataset that ships with scikit-learn, predicting
whether a patient's disease progresses above the cohort median from three
routine measurements — blood pressure and two blood-serum panels. This is a
genuinely hard task (AUC around 0.75), which is exactly where a confidence
band earns its keep: the uncertainty is real and worth quantifying.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_diabetes(as_frame=True)
X = data.data[["bp", "s3", "s4"]]
y = (data.target > data.target.median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=0
)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)
```

## One call

`from_estimator` mirrors scikit-learn's `RocCurveDisplay.from_estimator`: it
calls `predict_proba` (falling back to `decision_function`), selects the
positive-class column, and builds the band on held-out data.

```python
from rocci import from_estimator

band = from_estimator(clf, X_test, y_test, random_state=0)
band.plot(show_vacuous=True)
```

The shaded region contains the **entire** true population ROC curve with 95%
confidence — not each point separately, the whole curve at once. The hatched
sliver at the far left is the region where no distribution-free lower bound
exists (see below).

## Reading the summary

```python
print(band.summary())
```

Line by line:

- **coverage: 95% simultaneous** — the joint, whole-curve guarantee. You may
  read off as many operating points as you like without a multiplicity
  correction.
- **AUC + CI** — the exact Mann-Whitney AUC (identical to
  `sklearn.roc_auc_score`) with a bootstrap confidence interval.
- **band area** — mean vertical width; the tightness metric to compare
  settings or datasets.
- **the vacuous-region line** — below that FPR, certifying any lower bound
  is mathematically impossible without distributional assumptions; rocci
  says so instead of drawing one.

## Operating points

Suppose the screening context tolerates at most 10% false positives. What TPR
can we actually claim there?

```python
lower, tpr, upper = band.at(0.10)
print(f"at FPR = 10%:  TPR = {float(tpr):.3f},  "
      f"simultaneous bounds [{float(lower):.3f}, {float(upper):.3f}]")
```

Because the band is simultaneous, querying ten thresholds instead of one
costs nothing in validity.

## The band as data

```python
band.to_dataframe().head()
```

Everything on the result is a plain NumPy array (or scalar); `to_dataframe()`
is a convenience for the pandas-inclined. The `attribution` column records
which mechanism produced each point of the lower arm — the
[anatomy vignette](04-anatomy-of-the-band.md) explains how to read it.
