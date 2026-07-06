"""Public result objects: :class:`RocBand` and :class:`NormalityReport`.

Frozen dataclasses holding NumPy arrays: validation happens once at ingestion,
and the results are ndarray-centric, so a validation library would add a
dependency for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rocci._exceptions import RocciError
from rocci.band.grids import step_lookup

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import pandas as pd

    from rocci.band.envelope import EnvelopeBand

FloatArray = NDArray[np.float64]

#: Human-readable name per attribution code.
_ATTR_NAMES = {0: "bootstrap envelope", 1: "Beta floor", 2: "Wilson floor", 3: "pinned"}


@dataclass(frozen=True)
class NormalityReport:
    """Normality diagnostics for the Working-Hotelling band.

    Populated only on the ``normal=True`` path; ``RocBand`` carries ``None``
    on the envelope path. Each class gets two complementary checks —
    Shapiro-Francia (QQ-plot straightness, run for ``5 <= n <= 5000``) and
    D'Agostino K² (skewness + kurtosis, run for ``n >= 20``) — plus the moment
    effect sizes themselves; a check that does not apply at the class size (or
    to a constant class) reports ``nan`` and never creates suspicion. The
    ``suspect`` flag is the OR of every check: any one tripping flags the
    fit, with thresholds tuned to the MCC-optimal balance of sensitivity and
    specificity for predicting Working-Hotelling miscoverage. A quiet gate
    is weak evidence, not a certificate — there is no safe diagnostic
    region.

    Attributes:
        neg_sf_stat: Shapiro-Francia W' for the negative class.
        neg_sf_pvalue: Its p-value.
        neg_k2_stat: D'Agostino K² for the negative class.
        neg_k2_pvalue: Its p-value.
        neg_skew: Negative-class sample skewness (Gaussian ~ 0).
        neg_excess_kurtosis: Negative-class excess kurtosis (Gaussian ~ 0;
            positive = heavy tails, negative = short tails/bimodal).
        pos_sf_stat: Shapiro-Francia W' for the positive class.
        pos_sf_pvalue: Its p-value.
        pos_k2_stat: D'Agostino K² for the positive class.
        pos_k2_pvalue: Its p-value.
        pos_skew: Positive-class sample skewness.
        pos_excess_kurtosis: Positive-class excess kurtosis.
        probit_r2: OLS R² of probit-TPR on probit-FPR over the ROC interior
            (``nan`` when too few interior vertices).
        suspect: Whether binormality looks doubtful.
        warning: The exact warning text emitted (empty if none).

    Examples:
        >>> from rocci import NormalityReport
        >>> rep = NormalityReport(
        ...     neg_sf_stat=0.99,
        ...     neg_sf_pvalue=0.4,
        ...     neg_k2_stat=1.2,
        ...     neg_k2_pvalue=0.55,
        ...     neg_skew=0.05,
        ...     neg_excess_kurtosis=-0.1,
        ...     pos_sf_stat=0.98,
        ...     pos_sf_pvalue=0.3,
        ...     pos_k2_stat=2.0,
        ...     pos_k2_pvalue=0.37,
        ...     pos_skew=-0.02,
        ...     pos_excess_kurtosis=0.2,
        ...     probit_r2=0.995,
        ...     suspect=False,
        ...     warning="",
        ... )
        >>> rep.suspect
        False
        >>> rep.neg_pvalue  # headline: smallest check p-value for the class
        0.4
    """

    neg_sf_stat: float
    neg_sf_pvalue: float
    neg_k2_stat: float
    neg_k2_pvalue: float
    neg_skew: float
    neg_excess_kurtosis: float
    pos_sf_stat: float
    pos_sf_pvalue: float
    pos_k2_stat: float
    pos_k2_pvalue: float
    pos_skew: float
    pos_excess_kurtosis: float
    probit_r2: float
    suspect: bool
    warning: str

    @property
    def neg_pvalue(self) -> float:
        """Smallest negative-class check p-value (``nan`` if none applied).

        Falls below the suspect threshold exactly when one of the class's
        checks does — a headline number, not a calibrated single-test p-value.
        """
        return _min_pvalue(self.neg_sf_pvalue, self.neg_k2_pvalue)

    @property
    def pos_pvalue(self) -> float:
        """Smallest positive-class check p-value (``nan`` if none applied)."""
        return _min_pvalue(self.pos_sf_pvalue, self.pos_k2_pvalue)


def _min_pvalue(*pvalues: float) -> float:
    """Smallest non-NaN p-value, or NaN when no check applied."""
    valid = [p for p in pvalues if not np.isnan(p)]
    return min(valid) if valid else float("nan")


@dataclass(frozen=True)
class ScoreDiagnostics:
    """Sorted class scores retained for Working-Hotelling diagnostic plots.

    Internal payload of ``RocBand._diag`` on the ``normal=True`` path; holds
    references to (not copies of) the sorted score arrays the band was built
    from, so the QQ and probit-linearity panels of
    :meth:`RocBand.plot_diagnostics` can be drawn after the fact.

    Attributes:
        neg_sorted: Negative-class scores, ascending.
        pos_sorted: Positive-class scores, ascending.
    """

    neg_sorted: FloatArray
    pos_sorted: FloatArray


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
        auc: Exact Mann-Whitney AUC, ties weighted 1/2 — identical to
            ``sklearn.metrics.roc_auc_score``.
        auc_ci: Recentered percentile bootstrap AUC CI (``None`` for
            Working-Hotelling); consistent with ``auc`` even under ties.
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
    # Diagnostics payload for plot_diagnostics(): the EnvelopeBand
    # intermediates (envelope path) or the sorted scores (WH path). Internal —
    # K-sized arrays or references to arrays that already exist, never copies.
    _diag: EnvelopeBand | ScoreDiagnostics | None = field(
        default=None, repr=False, compare=False
    )

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

    def plot(
        self, ax: matplotlib.axes.Axes | None = None, **style: Any
    ) -> matplotlib.axes.Axes:
        """Plot the band, the empirical ROC, and the chance diagonal.

        Requires matplotlib (``pip install 'rocci[plot]'``); imported lazily.

        Args:
            ax: Axes to draw into; ``None`` creates a new figure.
            **style: Style overrides forwarded to
                :func:`rocci.plotting.plot_band` (``color``, ``band_alpha``,
                ``label``, ``show_vacuous``).

        Returns:
            The Axes, for composition.

        Raises:
            RocciError: If matplotlib is not installed.

        Examples:
            >>> import numpy as np
            >>> from rocci import roc_band
            >>> rng = np.random.default_rng(5)
            >>> y_true = np.r_[np.zeros(60), np.ones(60)]
            >>> y_score = np.r_[rng.normal(0, 1, 60), rng.normal(1.4, 1, 60)]
            >>> ax = roc_band(y_true, y_score, random_state=0).plot()
            >>> ax.get_xlabel()
            'False positive rate'
        """
        from rocci import plotting

        return plotting.plot_band(self, ax=ax, **style)

    def plot_diagnostics(
        self, fig: matplotlib.figure.Figure | None = None
    ) -> matplotlib.figure.Figure:
        """Plot the "why did my band do that" diagnostics figure.

        Envelope path: the band with the lower arm color-coded by floor
        attribution, plus the variance channels that drive the floor gate.
        Working-Hotelling path: the band plus the normality diagnostics
        (per-class QQ plots and the probit-linearity fit).

        Requires matplotlib (``pip install 'rocci[plot]'``); imported lazily.

        Args:
            fig: Figure to draw into; ``None`` creates a new one.

        Returns:
            The Figure.

        Raises:
            RocciError: If matplotlib is not installed.

        Examples:
            >>> import numpy as np
            >>> from rocci import roc_band
            >>> rng = np.random.default_rng(6)
            >>> y_true = np.r_[np.zeros(60), np.ones(60)]
            >>> y_score = np.r_[rng.normal(0, 1, 60), rng.normal(1.4, 1, 60)]
            >>> fig = roc_band(y_true, y_score, random_state=0).plot_diagnostics()
            >>> len(fig.axes)
            2
        """
        from rocci import plotting

        return plotting.plot_diagnostics(self, fig=fig)

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
