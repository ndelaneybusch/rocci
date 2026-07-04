"""Plot smoke tests (headless Agg) and the no-matplotlib error contract.

Risk mitigated: the plotting layer is the first thing users see; a figure
that silently drops the band, mislabels an axis, or crashes without
matplotlib breaks the five-line quickstart. Tests assert figures build
headless, every element is legend-labeled (vector-friendly contract), the
diagnostics panels match the band's method, and a missing matplotlib
produces the actionable install message.
"""

from __future__ import annotations

import builtins
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from rocci import roc_band
from rocci._exceptions import RocciError
from rocci.plotting import plot_band, plot_diagnostics
from tests.conftest import binormal_dataset


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def envelope_band():
    y_true, y_score = binormal_dataset(120, 120, seed=0)
    return roc_band(y_true, y_score, random_state=0)


@pytest.fixture(scope="module")
def wh_band():
    y_true, y_score = binormal_dataset(120, 120, seed=1)
    return roc_band(y_true, y_score, normal=True)


def legend_texts(ax):
    legend = ax.get_legend()
    assert legend is not None, "every panel must carry a legend"
    return [t.get_text() for t in legend.get_texts()]


class TestPlotBand:
    def test_returns_labeled_axes(self, envelope_band):
        ax = envelope_band.plot()
        assert ax.get_xlabel() == "False positive rate"
        assert ax.get_ylabel() == "True positive rate"
        texts = " ".join(legend_texts(ax))
        assert "95% simultaneous band (rocci envelope)" in texts
        assert "empirical ROC" in texts
        assert "chance" in texts

    def test_draws_into_existing_axes(self, envelope_band):
        _, ax = plt.subplots()
        returned = envelope_band.plot(ax=ax)
        assert returned is ax

    def test_style_overrides(self, envelope_band):
        ax = envelope_band.plot(color="#CC79A7", band_alpha=0.5, label="my band")
        assert "my band" in legend_texts(ax)

    def test_show_vacuous_adds_hatched_region(self, envelope_band):
        ax = plot_band(envelope_band, show_vacuous=True)
        assert any("no certifiable lower bound" in t for t in legend_texts(ax))

    def test_wh_band_annotation(self, wh_band):
        ax = wh_band.plot()
        assert any("Working-Hotelling" in t for t in legend_texts(ax))


class TestPlotDiagnostics:
    def test_envelope_two_panels(self, envelope_band):
        fig = envelope_band.plot_diagnostics()
        assert len(fig.axes) == 2
        band_ax, var_ax = fig.axes
        assert "attribution" in band_ax.get_title()
        assert var_ax.get_yscale() == "log"
        assert any("raw bootstrap variance" in t for t in legend_texts(var_ax))

    def test_envelope_floor_overlays_present(self):
        # high AUC + small n forces both floors to fire somewhere
        y_true, y_score = binormal_dataset(40, 40, auc=0.95, seed=2)
        band = roc_band(y_true, y_score, random_state=0)
        assert set(np.unique(band.attribution)) - {0, 3}, "fixture must floor"
        fig = band.plot_diagnostics()
        texts = " ".join(legend_texts(fig.axes[0]))
        assert "floor" in texts

    def test_wh_four_panels(self, wh_band):
        fig = wh_band.plot_diagnostics()
        assert len(fig.axes) == 4
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "QQ plot: negative class" in titles
        assert "QQ plot: positive class" in titles
        assert "probit-probit" in titles

    def test_draws_into_existing_figure(self, envelope_band):
        fig = plt.figure()
        returned = plot_diagnostics(envelope_band, fig=fig)
        assert returned is fig

    def test_band_without_payload_raises(self, envelope_band):
        import dataclasses

        stripped = dataclasses.replace(envelope_band, _diag=None)
        with pytest.raises(RocciError, match="diagnostics payload"):
            stripped.plot_diagnostics()


class TestNoMatplotlib:
    def test_actionable_error_without_matplotlib(self, envelope_band, monkeypatch):
        real_import = builtins.__import__

        def block_matplotlib(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError(f"no module named {name!r}")
            return real_import(name, *args, **kwargs)

        for mod in [m for m in sys.modules if m.startswith("matplotlib")]:
            monkeypatch.delitem(sys.modules, mod)
        monkeypatch.setattr(builtins, "__import__", block_matplotlib)
        with pytest.raises(RocciError, match=r"rocci\[plot\]"):
            envelope_band.plot()
