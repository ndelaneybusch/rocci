# rocci

[![PyPI](https://img.shields.io/pypi/v/rocci)](https://pypi.org/project/rocci/)
[![Python versions](https://img.shields.io/pypi/pyversions/rocci)](https://pypi.org/project/rocci/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/rocci)](https://anaconda.org/conda-forge/rocci)
[![CI](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml/badge.svg)](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml)
[![Merge gates](https://github.com/ndelaneybusch/rocci/actions/workflows/gates.yml/badge.svg)](https://github.com/ndelaneybusch/rocci/actions/workflows/gates.yml)
[![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ndelaneybusch/c2865a0da1db40e24976b7c721a4ca97/raw/rocci-coverage.json)](https://github.com/ndelaneybusch/rocci/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://ndelaneybusch.github.io/rocci)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/ndelaneybusch/rocci/blob/main/LICENSE)

**Distribution-free simultaneous confidence bands for ROC curves.**

<!-- Absolute URL so the figure renders on PyPI as well as GitHub. -->
![rocci envelope band vs Working-Hotelling band on heavy-tailed scores](https://raw.githubusercontent.com/ndelaneybusch/rocci/main/docs/assets/hero.png)

`rocci` is a simple interface to easily add uncertainty estimates to your ROC
curve that are very likely to be correct in nearly all use cases. It draws a
simultaneous confidence band, which maintains the specified confidence of
capturing the _entire_ true (population) ROC.

`rocci` is designed to:
- __just work__. It should do the right thing off the shelf for nearly any data
  set.
- __drop in to your workflow__. It natively integrates with sklearn, torch,
  statsmodels, PyMC/arviz, and pandas/polars data.
- __be fast__. Core operations are implemented in rust for speed (invisible to
  the user).
- __have a lightweight footprint__. Minimal runtime dependencies (just numpy and
  scipy).
- __support an open ecosystem__. Permissive MIT license, easy extensibility.

By default, `rocci` uses a distribution-free method that provides informative
bands without burdensome assumptions about the data. It maintains nominal
coverage in a huge variety of contexts and the rare violations tend to be small
misses. If you are comfortable adding a normality assumption to get tighter
bands, `rocci` yields a "Working-Hotelling" band, but also carefully checks the
normality assumption and warns you when it looks dicey.

## Installation

```bash
pip install rocci            # prebuilt wheels — no Rust toolchain needed
pip install 'rocci[plot]'    # + matplotlib for band.plot() and diagnostics
uv add rocci                 # in uv-managed projects
```

Wheels cover Linux (glibc x86-64/aarch64 and musl), macOS (Intel and Apple
silicon), and Windows, for every Python ≥ 3.10; runtime dependencies are
numpy and scipy only. On any other platform `pip` falls back to the sdist
(requires a [Rust toolchain](https://rustup.rs)), and if no compiled kernel
is present at runtime a pure-NumPy backend with identical statistical
semantics takes over automatically. A conda-forge package
(`conda install -c conda-forge rocci`) follows the first PyPI release.
Details: [installation guide](https://ndelaneybusch.github.io/rocci/getting-started/installation/).

## Quickstart

```python
from rocci import roc_band

band = roc_band(y_true, y_score)
band.plot()
print(band.summary())
```

Docs: <https://ndelaneybusch.github.io/rocci> ·
Changelog: [CHANGELOG.md](CHANGELOG.md) ·
Contributing (including the release process):
[CONTRIBUTING.md](CONTRIBUTING.md)
