# rocci

**Distribution-free simultaneous confidence bands for ROC curves.**

![Envelope band vs Working-Hotelling band on heavy-tailed scores](assets/hero.png)

`rocci` is a simple interface for adding uncertainty estimates to your ROC
curve. It draws a **simultaneous confidence band**, which maintains the
specified confidence of capturing the *entire* true (population) ROC — not
just each point one at a time. The default is a new nonparametric method
that works with nearly all common data distributions.

```python
from rocci import roc_band

band = roc_band(y_true, y_score)   # labels + scores, any common container
band.plot()
print(band.summary())
```

That is the whole quickstart. See it annotated in
[Getting started](getting-started/quickstart.md), or run the
[scikit-learn vignette](vignettes/01-quickstart-sklearn.md) end to end.

## Why rocci

`rocci` is designed to:

- **Just work.** It does the right thing off the shelf for nearly any data
  set: no normality assumption, safe under heavy ties and discrete scores,
  and honest where no distribution-free bound exists.
- **Drop in to your workflow.** Native integration with scikit-learn, torch,
  statsmodels, PyMC/arviz, and pandas/polars data — ingestion is duck-typed,
  with zero hard dependencies on any of those libraries.
- **Be fast.** The bootstrap kernel is compiled Rust (a pure-NumPy fallback
  keeps the package working everywhere): 2 000 bootstrap replicates on
  100 000 samples in well under half a second.
- **Make a minimal footprint.** The only hard dependency is numpy;
  plotting is an optional extra (`rocci[plot]`).
- **Clear an unreasonably high bar of rigor.** The method is
  [validated with millions of simulations across diverse data
  sets](method/simulations.md); the implementation is
  [verified with exacting tests](method/verification.md).
- **Support an open ecosystem.** Permissive MIT license, easy extensibility.

If you are comfortable adding a normality assumption to get a tighter band,
`normal=True` gives the parametric **Working–Hotelling** band — and rocci
checks the assumption and warns when it looks doubtful, as in the figure
above, where the true curve escapes the parametric band entirely.
[Which band should I use?](guide/which-band.md) explains the trade.

## Installation

```bash
pip install rocci            # prebuilt wheels (no need for rust toolchain)

# optional plotting support
pip install 'rocci[plot]'
```

Details in the [installation guide](getting-started/installation.md).

## Where to next

- [Reading the band](guide/reading-the-band.md) — what "simultaneous" buys
  you, and what the vacuous region at tiny FPR means.
- [The envelope method](method/envelope.md) — how the band is built.
- [Simulations and validation](method/simulations.md) — the evidence that the
  method works where the classical bands fail.
- [How rocci is verified](method/verification.md) — the case for trusting the
  numbers.
- [API reference](api.md) — the full public surface (it's small).

## Citing

If rocci contributes to a publication, please cite it — see
[`CITATION.cff`](https://github.com/ndelaneybusch/rocci/blob/main/CITATION.cff)
in the repository. `band.summary()` ends with the same pointer.
