# rocci

[![PyPI](https://img.shields.io/pypi/v/rocci)](https://pypi.org/project/rocci/)
[![Python versions](https://img.shields.io/pypi/pyversions/rocci)](https://pypi.org/project/rocci/)
[![CI](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml/badge.svg)](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml)
[![Merge gates](https://github.com/ndelaneybusch/rocci/actions/workflows/gates.yml/badge.svg)](https://github.com/ndelaneybusch/rocci/actions/workflows/gates.yml)
[![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ndelaneybusch/c2865a0da1db40e24976b7c721a4ca97/raw/rocci-coverage.json)](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://ndelaneybusch.github.io/rocci)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/ndelaneybusch/rocci/blob/main/LICENSE)

**Distribution-free simultaneous confidence bands for ROC curves.**

`rocci` is a simple interface to easily add uncertainty estimates to your ROC
curve. It draws a simultaneous confidence band, which maintains the specified
confidence of capturing the _entire_ true (population) ROC. `rocci` uses a new
nonparametric method by default that is likely to work with nearly all common
data distributions.

```python
from rocci import roc_band

band = roc_band(y_true, y_score)
band.plot()
print(band.summary())
```

## Why rocci

`rocci` is designed to:
- __just work__. It does the right thing off the shelf for nearly any data set.
- __drop in to your workflow__. It natively integrates with sklearn, torch,
  statsmodels, PyMC/arviz, and pandas/polars data.
- __be fast__. A rust backend blazes through the algorithm.
- __make a minimal footprint__. The only hard dependencies are numpy and scipy.
- __clear an unreasonably high bar of rigor__. Method is validated with
  [millions of simulations across diverse data
  sets](https://ndelaneybusch.github.io/rocci/method/simulations/),
  implementation is [verified with exacting
  tests](https://ndelaneybusch.github.io/rocci/latest/method/verification/).
- __support an open ecosystem__. Permissive MIT license, easy extensibility.

If you are comfortable adding a normality assumption to get tighter bands,
`rocci` yields the tighter "Working-Hotelling" band, but also carefully checks
the normality assumption and warns you when it looks dicey.

## Installation

```bash
pip install rocci            # prebuilt wheels (no need for rust toolchain)

# optional plotting support
pip install 'rocci[plot]'

```

Details in the [installation guide](https://ndelaneybusch.github.io/rocci/getting-started/installation/).

Docs: <https://ndelaneybusch.github.io/rocci> ·
Changelog: [CHANGELOG.md](CHANGELOG.md) ·
Contributing (including the release process):
[CONTRIBUTING.md](CONTRIBUTING.md)
