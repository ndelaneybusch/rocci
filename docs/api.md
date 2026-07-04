# API reference

The public surface is deliberately small: two band constructors, one
estimator convenience, the result object, and the warning/error taxonomy.
Everything else is private or documented as internal.

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
