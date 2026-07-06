"""Matplotlib visualization of :class:`~rocci._result.RocBand`.

matplotlib is an optional extra (``pip install 'rocci[plot]'``) and is imported
lazily inside each function, so importing :mod:`rocci.plotting` — or rocci
itself — never pays for it. House style: colorblind-safe Okabe-Ito palette, no
chartjunk, every element legend-labeled (vector-friendly), constrained layout.
Every routine takes an existing Axes/Figure and returns it for composition.

The attribution colors are fixed package-wide: yellow marks the Beta
order-statistic floor and green the Wilson rectangle floor, matching the
paper's floor-attribution graphics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rocci._exceptions import RocciError
from rocci._result import RocBand, ScoreDiagnostics
from rocci.band.envelope import ATTR_BETA_FLOOR, ATTR_WILSON_FLOOR, EnvelopeBand
from rocci.band.grids import empirical_roc_vertices
from rocci.band.normal import _INTERIOR_HI, _INTERIOR_LO
from rocci.special import ndtri

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

__all__ = ["plot_band", "plot_diagnostics"]

#: Okabe-Ito colorblind-safe palette (fixed house style).
_BLUE = "#0072B2"  # band + empirical ROC
_YELLOW = "#E69F00"  # Beta order-statistic floor
_GREEN = "#009E73"  # Wilson rectangle floor
_GREY = "#7F7F7F"  # chance diagonal, reference lines
_FIGSIZE = (5.5, 5.0)


def _require_pyplot():
    """Import pyplot lazily, or raise the actionable install message."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise RocciError(
            "plotting requires matplotlib — pip install 'rocci[plot]'"
        ) from err
    return plt


def _runs(mask: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Contiguous ``True`` runs of ``mask`` as ``(start, stop)`` index pairs.

    ``stop`` is exclusive, so ``mask[start:stop]`` is all ``True``.
    """
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return edges.reshape(-1, 2)


def plot_band(
    band: RocBand,
    ax: Axes | None = None,
    *,
    color: str = _BLUE,
    band_alpha: float = 0.25,
    label: str | None = None,
    show_vacuous: bool = False,
) -> Axes:
    """Plot the confidence band, the empirical ROC, and the chance diagonal.

    Both the band arms and the empirical curve are drawn as right-continuous
    steps — the same convention used by :meth:`~rocci.RocBand.at` — so
    the figure shows exactly what the band certifies.

    Args:
        band: The band to plot.
        ax: Axes to draw into; ``None`` creates a ``(5.5, 5.0)`` figure.
        color: Band and ROC line color.
        band_alpha: Opacity of the band fill.
        label: Legend label for the band; ``None`` composes
            ``"{confidence:.0%} simultaneous band (<method>)"``.
        show_vacuous: Hatch the FPR region below ``band.vacuous_below``
            where no distribution-free lower bound exists.

    Returns:
        The Axes, for composition.

    Raises:
        RocciError: If matplotlib is not installed.

    Examples:
        >>> import numpy as np
        >>> from rocci import roc_band
        >>> from rocci.plotting import plot_band
        >>> rng = np.random.default_rng(0)
        >>> y = np.r_[np.zeros(60), np.ones(60)]
        >>> s = np.r_[rng.normal(0, 1, 60), rng.normal(1.5, 1, 60)]
        >>> ax = plot_band(roc_band(y, s, random_state=0), show_vacuous=True)
        >>> ax.get_ylabel()
        'True positive rate'
    """
    plt = _require_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=_FIGSIZE, layout="constrained")

    method_note = "rocci envelope" if band.method == "envelope" else "Working-Hotelling"
    band_label = (
        label
        if label is not None
        else f"{band.confidence:.0%} simultaneous band ({method_note})"
    )
    ax.fill_between(
        band.fpr,
        band.lower,
        band.upper,
        step="post",
        color=color,
        alpha=band_alpha,
        linewidth=0,
        label=band_label,
    )
    ax.plot(
        band.fpr,
        band.tpr,
        drawstyle="steps-post",
        color=color,
        linewidth=1.5,
        label=f"empirical ROC (AUC = {band.auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", color=_GREY, linewidth=1.0, label="chance")
    if show_vacuous and band.vacuous_below is not None and band.vacuous_below > 0.0:
        ax.axvspan(
            0.0,
            band.vacuous_below,
            facecolor="none",
            edgecolor=_GREY,
            hatch="///",
            linewidth=0,
            label=f"no certifiable lower bound (FPR < {band.vacuous_below:.3g})",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize="small")
    return ax


def plot_diagnostics(band: RocBand, fig: Figure | None = None) -> Figure:
    """Render the two-panel "why did my band do that here" figure.

    Envelope path: panel 1 is the band with the lower arm color-coded by
    floor attribution (yellow Beta floor, green Wilson rectangle floor) and
    jurisdiction boundaries marked; panel 2 shows the variance channels that
    drive the floor gate (raw bootstrap variance vs the Wilson floor, log
    scale) with the active-floor regions shaded.

    Working-Hotelling path: the band plus the normality evidence — per-class
    normal QQ plots and the probit-probit ROC linearity fit with its R².

    Args:
        band: A band produced by :func:`rocci.roc_band`.
        fig: Figure to draw into; ``None`` creates one.

    Returns:
        The Figure.

    Raises:
        RocciError: If matplotlib is not installed, or if ``band`` carries no
            diagnostics payload (i.e. it was not produced by ``roc_band``).

    Examples:
        >>> import numpy as np
        >>> from rocci import roc_band
        >>> from rocci.plotting import plot_diagnostics
        >>> rng = np.random.default_rng(1)
        >>> y = np.r_[np.zeros(60), np.ones(60)]
        >>> s = np.r_[rng.normal(0, 1, 60), rng.normal(1.5, 1, 60)]
        >>> fig = plot_diagnostics(roc_band(y, s, random_state=0))
        >>> len(fig.axes)
        2
    """
    plt = _require_pyplot()
    diag = band._diag
    if isinstance(diag, EnvelopeBand):
        if fig is None:
            fig = plt.figure(
                figsize=(2 * _FIGSIZE[0], _FIGSIZE[1]), layout="constrained"
            )
        ax_band, ax_var = fig.subplots(1, 2)
        _draw_attribution_panel(band, ax_band)
        _draw_variance_panel(band, diag, ax_var)
        return fig
    if isinstance(diag, ScoreDiagnostics):
        if fig is None:
            fig = plt.figure(
                figsize=(2 * _FIGSIZE[0], 2 * _FIGSIZE[1]), layout="constrained"
            )
        axes = fig.subplots(2, 2)
        plot_band(band, ax=axes[0, 0])
        _draw_probit_panel(band, diag, axes[0, 1])
        _draw_qq_panel(diag.neg_sorted, "negative class", axes[1, 0])
        _draw_qq_panel(diag.pos_sorted, "positive class", axes[1, 1])
        return fig
    raise RocciError(
        "this RocBand carries no diagnostics payload; bands built by "
        "rocci.roc_band always do. Rebuild the band with roc_band to plot "
        "diagnostics."
    )


def _draw_attribution_panel(band: RocBand, ax: Axes) -> None:
    """Band plot with the lower arm color-coded by floor attribution."""
    plot_band(band, ax=ax, show_vacuous=True)
    for code, color, name in (
        (ATTR_BETA_FLOOR, _YELLOW, "Beta floor"),
        (ATTR_WILSON_FLOOR, _GREEN, "Wilson floor"),
    ):
        mask = band.attribution == code
        if not mask.any():
            continue
        segment = np.where(mask, band.lower, np.nan)
        ax.plot(
            band.fpr,
            segment,
            drawstyle="steps-post",
            color=color,
            linewidth=2.5,
            label=f"lower bound: {name}",
        )
        for start, stop in _runs(mask):  # jurisdiction boundaries
            ax.axvline(band.fpr[start], color=color, linewidth=0.8, alpha=0.6)
            ax.axvline(band.fpr[stop - 1], color=color, linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right", fontsize="small")
    ax.set_title("band with lower-arm floor attribution")


def _draw_variance_panel(band: RocBand, env: EnvelopeBand, ax: Axes) -> None:
    """Variance channels vs FPR: why the rectangle floor fired where it did."""
    ax.plot(
        band.fpr,
        env.var_raw,
        color=_BLUE,
        linewidth=1.5,
        label="raw bootstrap variance",
    )
    ax.plot(
        band.fpr,
        env.wilson_var,
        color=_GREY,
        linestyle="--",
        linewidth=1.5,
        label="Wilson variance floor",
    )
    for code, color, name in (
        (ATTR_BETA_FLOOR, _YELLOW, "Beta floor active"),
        (ATTR_WILSON_FLOOR, _GREEN, "Wilson floor active"),
    ):
        runs = _runs(band.attribution == code)
        for i, (start, stop) in enumerate(runs):
            ax.axvspan(
                band.fpr[start],
                band.fpr[stop - 1],
                color=color,
                alpha=0.15,
                linewidth=0,
                label=name if i == 0 else None,
            )
    ax.set_yscale("log")  # nonpositive values (degenerate cells) are masked
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("TPR variance (log scale)")
    ax.set_title("variance channels and floor jurisdictions")
    ax.legend(loc="lower right", fontsize="small")


def _draw_qq_panel(scores: NDArray[np.float64], name: str, ax: Axes) -> None:
    """Normal QQ plot of one class's scores, with a moment-matched reference line."""
    n = len(scores)
    theoretical = ndtri((np.arange(1, n + 1) - 0.5) / n)
    ax.plot(
        theoretical,
        scores,
        linestyle="none",
        marker=".",
        markersize=3,
        color=_BLUE,
        label="score quantiles",
    )
    mu, sd = float(np.mean(scores)), float(np.std(scores, ddof=1))
    ax.plot(
        theoretical,
        mu + sd * theoretical,
        color=_GREY,
        linestyle="--",
        linewidth=1.0,
        label="normal reference",
    )
    ax.set_xlabel("normal quantiles")
    ax.set_ylabel("score quantiles")
    ax.set_title(f"QQ plot: {name}")
    ax.legend(loc="lower right", fontsize="small")


def _draw_probit_panel(band: RocBand, diag: ScoreDiagnostics, ax: Axes) -> None:
    """Probit-probit ROC linearity: straight iff the binormal model holds."""
    fpr_v, tpr_v = empirical_roc_vertices(diag.neg_sorted, diag.pos_sorted)
    interior = (
        (fpr_v > _INTERIOR_LO)
        & (fpr_v < _INTERIOR_HI)
        & (tpr_v > _INTERIOR_LO)
        & (tpr_v < _INTERIOR_HI)
    )
    pts = np.unique(np.column_stack([fpr_v[interior], tpr_v[interior]]), axis=0)
    r2 = band.normality.probit_r2 if band.normality is not None else float("nan")
    if len(pts) >= 2:
        xi, yi = ndtri(pts[:, 0]), ndtri(pts[:, 1])
        ax.plot(
            xi,
            yi,
            linestyle="none",
            marker=".",
            markersize=4,
            color=_BLUE,
            label="ROC interior vertices",
        )
        slope, intercept = np.polyfit(xi, yi, 1)
        xx = np.array([np.min(xi), np.max(xi)])
        ax.plot(
            xx,
            slope * xx + intercept,
            color=_GREY,
            linestyle="--",
            linewidth=1.0,
            label=f"OLS fit (R² = {r2:.3f})",
        )
        ax.legend(loc="lower right", fontsize="small")
    else:
        ax.text(
            0.5,
            0.5,
            "too few interior vertices\nfor the probit-linearity check",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=_GREY,
        )
    ax.set_xlabel("probit(FPR)")
    ax.set_ylabel("probit(TPR)")
    ax.set_title("probit-probit ROC linearity")
