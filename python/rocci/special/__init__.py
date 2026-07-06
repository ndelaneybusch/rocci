"""Self-contained special functions and normality statistics for the band core.

The band machinery needs a small, fixed set of distributional quantities:
normal quantiles and CDF values, Beta quantiles at integer shape parameters
(order-statistic distributions), the chi-squared(df=2) quantile, and the
normality checks behind the Working-Hotelling diagnostics (Shapiro-Francia,
D'Agostino K², and the moment effect sizes they are built from). This package
implements exactly that set — restricted to the cases rocci actually
evaluates — so the statistical core does not depend on a general
special-functions library.

Everything with a SciPy counterpart is validated against it in
``tests/test_special.py`` to near machine precision on the supported domains;
Shapiro-Francia (which SciPy lacks) is validated there by Monte-Carlo null
calibration and power checks instead.
"""

from rocci.special._beta import beta_ppf
from rocci.special._chi2 import chi2_ppf
from rocci.special._dagostino import dagostino_k2, skew_kurtosis
from rocci.special._normal import ndtr, ndtri
from rocci.special._shapiro_francia import shapiro_francia

__all__ = [
    "beta_ppf",
    "chi2_ppf",
    "dagostino_k2",
    "ndtr",
    "ndtri",
    "shapiro_francia",
    "skew_kurtosis",
]
