"""Distribution-free simultaneous confidence bands for ROC curves.

The public API (``roc_band``, ``roc_band_ovr``, ``from_estimator``,
``RocBand``, ``show_versions``) lands in milestone M3. Until then the
package exposes the statistical core under private modules and the
bootstrap backend under :mod:`rocci.backend`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rocci")
except PackageNotFoundError:  # pragma: no cover — source tree without install
    __version__ = "0+unknown"

__all__ = ["__version__"]
