"""Distribution-free simultaneous confidence bands for ROC curves.

The public surface is small by design: :func:`roc_band` (and its multiclass
sibling :func:`roc_band_ovr`), the :func:`from_estimator` convenience, the
:class:`RocBand` result object, and :func:`show_versions` for bug reports.
Everything else is private (``rocci._*``) or documented as internal.
"""

from importlib.metadata import PackageNotFoundError, version

from rocci._api import from_estimator, roc_band, roc_band_ovr, show_versions
from rocci._result import NormalityReport, RocBand

try:
    __version__ = version("rocci")
except PackageNotFoundError:  # pragma: no cover — source tree without install
    __version__ = "0+unknown"

__all__ = [
    "NormalityReport",
    "RocBand",
    "__version__",
    "from_estimator",
    "roc_band",
    "roc_band_ovr",
    "show_versions",
]
