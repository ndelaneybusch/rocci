# rocci

[![PyPI](https://img.shields.io/pypi/v/rocci)](https://pypi.org/project/rocci/)
[![Python versions](https://img.shields.io/pypi/pyversions/rocci)](https://pypi.org/project/rocci/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/rocci)](https://anaconda.org/conda-forge/rocci)
[![CI](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml/badge.svg)](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml)
[![Merge gates](https://github.com/ndelaneybusch/rocci/actions/workflows/gates.yml/badge.svg)](https://github.com/ndelaneybusch/rocci/actions/workflows/gates.yml)
[![codecov](https://codecov.io/gh/ndelaneybusch/rocci/branch/main/graph/badge.svg)](https://codecov.io/gh/ndelaneybusch/rocci)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://ndelaneybusch.github.io/rocci)
[![License](https://img.shields.io/pypi/l/rocci)](https://github.com/ndelaneybusch/rocci/blob/main/LICENSE)

**Distribution-free simultaneous confidence bands for ROC curves.**

`rocci` implements the studentized bootstrap envelope method: a simultaneous
confidence band for the true population ROC curve that is calibrated across
score distributions, honest at the extremes (exact Wilson and Beta
order-statistic floors), and fast (pure-Rust bootstrap kernel with a NumPy
fallback).

```python
import rocci

band = rocci.roc_band(y_true, y_score)   # lands in milestone M3
band.plot()
print(band.summary())
```

> **Status**: pre-release. The statistical core (empirical ROC, floors,
> envelope assembly, golden-master equivalence to the validated paper
> implementation) and the Rust bootstrap kernel are complete; the public
> `roc_band` API, Working–Hotelling path, plotting, and docs site are in
> progress. See `CHANGELOG.md`.

**When to use this**: whenever you would draw an ROC curve from a finite
sample and want the uncertainty of the *whole curve* — not a pointwise CI at
one operating point, and not a parametric band that silently fails when
scores aren't binormal. The band is invariant to monotone transforms of the
scores (logits vs probabilities give identical bands) and remains valid
under heavy ties.

Docs (once published): <https://ndelaneybusch.github.io/rocci> · Contributing:
[CONTRIBUTING.md](CONTRIBUTING.md)
