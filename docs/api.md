# API reference

The public surface is deliberately small: two band constructors, one
estimator convenience, the result object, and the warning/error taxonomy.
Everything else is private or documented as internal.

## At a glance

| Object | Description |
| --- | --- |
| [`roc_band`][rocci.roc_band] | Simultaneous confidence band for a binary ROC curve |
| [`roc_band_ovr`][rocci.roc_band_ovr] | One-vs-rest bands for multiclass scores |
| [`from_estimator`][rocci.from_estimator] | Build a band directly from a fitted scikit-learn-style estimator |
| [`RocBand`][rocci.RocBand] | Frozen result object: band arrays, AUC, plotting, export |
| [`NormalityReport`][rocci.NormalityReport] | Diagnostics attached when `normal=True` |
| [`plotting.plot_band`][rocci.plotting.plot_band] | Free-function band plot for composing your own figures |
| [`plotting.plot_diagnostics`][rocci.plotting.plot_diagnostics] | Free-function diagnostic panels |
| [`show_versions`][rocci.show_versions] | Environment report for bug reports |

All rocci warnings derive from [`RocciWarning`][rocci.RocciWarning] and all
input errors are [`RocciError`][rocci.RocciError], so one
`filterwarnings` rule or `except` clause covers the whole family.

## Band constructors

::: rocci.roc_band

::: rocci.roc_band_ovr

::: rocci.from_estimator

## Results

::: rocci.RocBand

::: rocci.NormalityReport

## Plotting

Free-function equivalents of the `RocBand` plot methods, for composition into
your own figures. matplotlib is an optional extra (`pip install
'rocci[plot]'`).

::: rocci.plotting.plot_band

::: rocci.plotting.plot_diagnostics

## Warnings and errors

All rocci warnings derive from `RocciWarning`, so one `filterwarnings` rule
silences or escalates the whole family; all input errors are `RocciError`
(a `ValueError` subclass).

::: rocci.RocciError

::: rocci.RocciWarning

::: rocci.NormalityWarning

::: rocci.LowConfidenceWarning

::: rocci.SmallSampleWarning

::: rocci.TiesWarning

::: rocci.FallbackBackendWarning

## Environment

::: rocci.show_versions
