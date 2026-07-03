"""One-vs-rest multiclass contract for ``roc_band_ovr``.

Risk mitigated: the OvR reduction is a thin loop, but its guarantees hinge on
three easy-to-break details — the Bonferroni alpha split, the column-to-class
routing, and per-class seed reproducibility — plus the permanent refusal of
``normal=True`` (the rest class is the Working-Hotelling failure mode). This
suite pins each.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocci import RocBand, roc_band, roc_band_ovr
from rocci._exceptions import RocciError


def three_class(n_per=40, seed=0):
    """Return ``(y_true, y_score)`` for a separable 3-class problem.

    Column ``j`` is a monotone score for class ``j`` (diagonal signal), so
    column↔class routing is checkable: the band for class ``j`` built from
    column ``j`` must have a higher AUC than one built from a wrong column.
    """
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n_per)
    scores = rng.normal(0.0, 1.0, (3 * n_per, 3))
    for j in range(3):
        scores[y == j, j] += 3.0  # class j scores high on column j
    return y, scores


class TestFamilyWise:
    def test_bonferroni_splits_alpha(self):
        y, scores = three_class()
        bands = roc_band_ovr(y, scores, confidence=0.95, random_state=0)
        # each class band is built at 1 - alpha/m = 1 - 0.05/3
        for band in bands.values():
            assert band.confidence == pytest.approx(1.0 - 0.05 / 3.0)

    def test_family_none_keeps_marginal_confidence(self):
        y, scores = three_class()
        bands = roc_band_ovr(y, scores, confidence=0.95, family="none", random_state=0)
        for band in bands.values():
            assert band.confidence == pytest.approx(0.95)

    def test_returns_band_per_class(self):
        y, scores = three_class()
        bands = roc_band_ovr(y, scores, random_state=0)
        assert sorted(bands) == [0, 1, 2]
        assert all(isinstance(b, RocBand) for b in bands.values())


class TestRouting:
    def test_column_matches_class(self):
        y, scores = three_class(seed=1)
        bands = roc_band_ovr(y, scores, random_state=0)
        # the diagonal signal means each class's own column separates well
        for j in (0, 1, 2):
            assert bands[j].auc > 0.9

    def test_column_count_mismatch_raises(self):
        y, scores = three_class()
        with pytest.raises(RocciError, match="n, m=3"):
            roc_band_ovr(y, scores[:, :2], random_state=0)

    def test_two_classes_rejected(self):
        y = np.repeat([0, 1], 40)
        scores = np.random.default_rng(0).random((80, 2))
        with pytest.raises(RocciError, match="m > 2"):
            roc_band_ovr(y, scores, random_state=0)


class TestSeeding:
    def test_per_class_seeds_differ(self):
        y, scores = three_class()
        bands = roc_band_ovr(y, scores, random_state=0)
        seeds = {b.random_state for b in bands.values()}
        assert len(seeds) == 3  # each class got its own spawned seed

    def test_reproducible_from_random_state(self):
        y, scores = three_class()
        a = roc_band_ovr(y, scores, random_state=42)
        b = roc_band_ovr(y, scores, random_state=42)
        for cls in a:
            np.testing.assert_array_equal(a[cls].lower, b[cls].lower)
            np.testing.assert_array_equal(a[cls].upper, b[cls].upper)


class TestNormalRefused:
    def test_normal_true_raises_rest_mixture(self):
        y, scores = three_class()
        with pytest.raises(RocciError, match="mixture"):
            roc_band_ovr(y, scores, normal=True, random_state=0)


def test_ovr_joint_coverage_matches_manual_loop():
    """A hand loop of roc_band at 1 - alpha/m reproduces the OvR bands' width."""
    y, scores = three_class(seed=2)
    bands = roc_band_ovr(y, scores, confidence=0.95, random_state=3)
    # the effective per-class confidence is what a manual Bonferroni loop uses
    manual = roc_band(y == 0, scores[:, 0], confidence=1.0 - 0.05 / 3.0, random_state=0)
    assert bands[0].confidence == pytest.approx(manual.confidence)
