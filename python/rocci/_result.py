"""Public result objects: :class:`RocBand` and :class:`NormalityReport`.

Frozen dataclasses holding NumPy arrays: validation happens once at ingestion,
and the results are ndarray-centric, so a validation library would add a
dependency for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rocci._exceptions import RocciError
from rocci.band.grids import step_lookup

if TYPE_CHECKING:
    import pandas as pd

FloatArray = NDArray[np.float64]

#: Human-readable name per attribution code.
_ATTR_NAMES = {0: "bootstrap envelope", 1: "Beta floor", 2: "Wilson floor", 3: "pinned"}


@dataclass(frozen=True)
class NormalityReport:
    """Normality diagnostics for the Working-Hotelling band.

    Populated only on the ``normal=True`` path; ``RocBand`` carries ``None``
    on the envelope path.

    Attributes:
        neg_test: Test used on the negative class (``"shapiro"`` / ``"normaltest"``).
        neg_stat: Its statistic.
        neg_pvalue: Its p-value.
        pos_test: Test used on the positive class.
        pos_stat: Its statistic.
        pos_pvalue: Its p-value.
        probit_r2: OLS R² of probit-TPR on probit-FPR over the ROC interior
            (``nan`` when too few interior vertices).
        suspect: Whether binormality looks doubtful.
        warning: The exact warning text emitted (empty if none).

    Examples:
        >>> from rocci import NormalityReport
        >>> rep = NormalityReport(
        ...     "shapiro", 0.99, 0.4, "shapiro", 0.98, 0.3, 0.995, False, ""
        ... )
        >>> rep.suspect
        False
    """

    neg_test: str
    neg_stat: float
    neg_pvalue: float
    pos_test: str
    pos_stat: float
    pos_pvalue: float
    probit_r2: float
    suspect: bool
    warning: str


@dataclass(frozen=True, eq=False)
class RocBand:
    """A simultaneous confidence band for a population ROC curve.

    Returned by :func:`rocci.roc_band`. Immutable; arrays are shape ``(K,)`` on
    the FPR grid ``fpr``. ``eq`` is disabled because array fields make
    value-equality ill-defined.

    Attributes:
        fpr: FPR grid, shape ``(K,)``.
        tpr: Empirical ROC at the grid.
        lower: Lower band.
        upper: Upper band.
        confidence: Simultaneous coverage level of this band.
        method: ``"envelope"`` or ``"working_hotelling"``.
        n_neg: Number of negatives.
        n_pos: Number of positives.
        n_boot: Bootstrap replicates (``None`` for Working-Hotelling).
        auc: Trapezoid AUC on the full empirical ROC (not the grid).
        auc_ci: Percentile bootstrap AUC CI (``None`` for Working-Hotelling).
        attribution: int8 codes — 0 bootstrap, 1 Beta floor, 2 Wilson floor,
            3 pinned endpoint.
        vacuous_below: FPR below which the lower band is provably vacuous
            (``None`` for Working-Hotelling).
        normality: Diagnostics on the ``normal=True`` path, else ``None``.
        backend: Compute backend that produced the band (``"rust"``/``"numpy"``).
        random_state: The seed the caller passed (``None`` if unseeded).
        notes: INFO-level ingestion notes surfaced by :meth:`summary`.

    Examples:
        >>> import numpy as np
        >>> from rocci import roc_band
        >>> rng = np.random.default_rng(0)
        >>> y_true = np.r_[np.zeros(60), np.ones(60)]
        >>> y_score = np.r_[rng.normal(0, 1, 60), rng.normal(1.4, 1, 60)]
        >>> band = roc_band(y_true, y_score, random_state=0)
        >>> band.fpr.shape == band.lower.shape == band.upper.shape
        True
        >>> bool((band.lower <= band.upper + 1e-12).all())
        True
    """

    fpr: FloatArray
    tpr: FloatArray
    lower: FloatArray
    upper: FloatArray
    confidence: float
    method: Literal["envelope", "working_hotelling"]
    n_neg: int
    n_pos: int
    n_boot: int | None
    auc: float
    auc_ci: tuple[float, float] | None
    attribution: NDArray[np.int8]
    vacuous_below: float | None
    normality: NormalityReport | None
    backend: Literal["rust", "numpy"]
    random_state: int | None
    notes: tuple[str, ...] = ()

    @property
    def band_area(self) -> float:
        """Mean vertical band width — a scalar tightness metric for the band.

        Examples:
            >>> import numpy as np
            >>> from rocci import roc_band
            >>> rng = np.random.default_rng(1)
            >>> y_true = np.r_[np.zeros(60), np.ones(60)]
            >>> y_score = np.r_[rng.normal(0, 1, 60), rng.normal(1.4, 1, 60)]
            >>> band = roc_band(y_true, y_score, random_state=0)
            >>> 0.0 < band.band_area < 1.0
            True
        """
        return float(np.mean(self.upper - self.lower))

    def at(self, fpr: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Step-interpolate the band at arbitrary FPR points.

        Uses the same right-continuous step convention as the band construction.

        Args:
            fpr: Query FPR value(s) in ``[0, 1]``.

        Returns:
            Tuple ``(lower, tpr, upper)`` at the query points.

        Raises:
            RocciError: If any query is outside ``[0, 1]``.

        Examples:
            >>> import numpy as np
            >>> from rocci import roc_band
            >>> rng = np.random.default_rng(2)
            >>> y_true = np.r_[np.zeros(60), np.ones(60)]
            >>> y_score = np.r_[rng.normal(0, 1, 60), rng.normal(1.4, 1, 60)]
            >>> band = roc_band(y_true, y_score, random_state=0)
            >>> lo, tp, up = band.at([0.1, 0.5])
            >>> bool((lo <= tp).all() and (tp <= up).all())
            True
        """
        query = np.asarray(fpr, dtype=np.float64)
        # NaN fails every comparison, so check containment as a negation.
        if query.size and not ((query >= 0.0) & (query <= 1.0)).all():
            raise RocciError(
                "at() queries must lie in [0, 1] (the FPR axis); got a value "
                "outside that range or NaN."
            )
        return (
            step_lookup(self.fpr, self.lower, query),
            step_lookup(self.fpr, self.tpr, query),
            step_lookup(self.fpr, self.upper, query),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the band as a pandas DataFrame (lazy import).

        Returns:
            Columns ``[fpr, lower, tpr, upper, attribution]``.

        Raises:
            RocciError: If pandas is not installed.

        Examples:
            >>> import numpy as np
            >>> from rocci import roc_band
            >>> rng = np.random.default_rng(3)
            >>> y_true = np.r_[np.zeros(60), np.ones(60)]
            >>> y_score = np.r_[rng.normal(0, 1, 60), rng.normal(1.4, 1, 60)]
            >>> band = roc_band(y_true, y_score, random_state=0)
            >>> list(band.to_dataframe().columns)
            ['fpr', 'lower', 'tpr', 'upper', 'attribution']
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise RocciError(
                "to_dataframe() requires pandas — pip install pandas."
            ) from err
        return pd.DataFrame(
            {
                "fpr": self.fpr,
                "lower": self.lower,
                "tpr": self.tpr,
                "upper": self.upper,
                "attribution": self.attribution,
            }
        )

    def summary(self) -> str:
        """Return a human-readable report of the band.

        Examples:
            >>> import numpy as np
            >>> from rocci import roc_band
            >>> rng = np.random.default_rng(4)
            >>> y_true = np.r_[np.zeros(60), np.ones(60)]
            >>> y_score = np.r_[rng.normal(0, 1, 60), rng.normal(1.4, 1, 60)]
            >>> text = roc_band(y_true, y_score, random_state=0).summary()
            >>> text.startswith("rocci confidence band (envelope)")
            True
        """
        method_label = "envelope" if self.method == "envelope" else "Working-Hotelling"
        lines = [
            f"rocci confidence band ({method_label})",
            f"  samples: n_neg={self.n_neg}, n_pos={self.n_pos}",
            f"  coverage: {self.confidence:.0%} simultaneous",
            f"  AUC: {self.auc:.4f}"
            + (
                f"  (CI: {self.auc_ci[0]:.4f}, {self.auc_ci[1]:.4f})"
                if self.auc_ci is not None
                else ""
            ),
            f"  band area (mean width): {self.band_area:.4f}",
            f"  backend: {self.backend}"
            + (f", n_boot={self.n_boot}" if self.n_boot is not None else ""),
        ]
        if self.vacuous_below is not None:
            lines.append(
                f"  no distribution-free lower bound exists below FPR ~= "
                f"{self.vacuous_below:.4f}; increase the number of negatives to "
                "certify lower FPRs."
            )
        jurisdictions = {
            _ATTR_NAMES[code]: int(np.count_nonzero(self.attribution == code))
            for code in (1, 2)
        }
        active = [f"{name}: {n} pts" for name, n in jurisdictions.items() if n]
        if active:
            lines.append("  floor jurisdictions: " + ", ".join(active))
        if self.normality is not None and self.normality.suspect:
            lines.append("  normality: SUSPECT - " + self.normality.warning)
        lines.extend(f"  note: {n}" for n in self.notes)
        lines.append("  please cite rocci (see CITATION.cff).")
        return "\n".join(lines)
