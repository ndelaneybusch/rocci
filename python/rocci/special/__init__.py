"""Self-contained special functions and normality statistics for the band core.

The band machinery needs a small, fixed set of distributional quantities:
normal quantiles and CDF values, Beta quantiles at integer shape parameters
(order-statistic distributions), the chi-squared(df=2) quantile, and the
D'Agostino K² normality test. This package implements exactly that set —
restricted to the cases rocci actually evaluates — so the statistical core
does not depend on a general special-functions library.

Every function is validated against its SciPy counterpart in
``tests/test_special.py``; the implementations agree to near machine
precision on the supported domains.
"""

from rocci.special._beta import beta_ppf
from rocci.special._chi2 import chi2_ppf
from rocci.special._dagostino import dagostino_k2
from rocci.special._normal import ndtr, ndtri

__all__ = ["beta_ppf", "chi2_ppf", "dagostino_k2", "ndtr", "ndtri"]
