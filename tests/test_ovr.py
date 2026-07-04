"""One-vs-rest multiclass contract for ``roc_band_ovr``.

``roc_band_ovr`` is a thin loop over ``roc_band``, but "thin" is where silent
mis-wiring hides: a band can look perfect while scoring the wrong class, using an
uncorrected alpha, or reusing a seed. The signal in the fixtures is diagonal —
column ``j`` separates class ``j`` — so a routing slip craters that class's AUC
and is caught rather than hidden.

Guaranteed. The family-wise correction is exact: each per-class band is built at
``1 - alpha/m`` under the default Bonferroni split and at the marginal level under
``family="none"``. Column-to-class routing follows the ``classes`` key, not the
sorted-unique order (checked with a permuted order and string labels), and each
class gets its own spawned seed so runs are reproducible from ``random_state``.
The strongest check rebuilds the entire reduction independently from the spec 3.6
contract — one-vs-rest labels, column selection, per-class confidence and seeds —
and demands bit-identical band arrays, so a wrong column, an alpha that is
displayed but not actually used, or a drifted seed derivation each fail. Malformed
calls (bad ``family``, column-count mismatch, ``m <= 2``, a ``classes`` entry
absent from ``y_true``, duplicate classes) raise naming the problem, and
``normal=True`` is permanently refused because the rest class is a mixture.

Limitations. This certifies the reduction wiring; the correctness of each
underlying band is the single-class suites' job. Reproducibility inherits the
same fixed-backend/version caveat as ``roc_band``.
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

    def test_classes_order_routes_columns(self):
        # `classes` is the column key: column j must score classes[j], not
        # the sorted-unique order. Build the diagonal signal against a
        # permuted order and string labels; a routing slip would send a
        # near-random column to some class and crater its AUC.
        rng = np.random.default_rng(4)
        y = np.repeat(np.array(["ant", "bee", "cat"]), 40)
        order = ["cat", "ant", "bee"]
        scores = rng.normal(0.0, 1.0, (120, 3))
        for j, cls in enumerate(order):
            scores[y == cls, j] += 3.0
        bands = roc_band_ovr(y, scores, classes=order, random_state=0)
        assert list(bands) == order
        for cls in order:
            assert bands[cls].auc > 0.9, f"column routed to the wrong class: {cls}"

    def test_kwargs_forwarded_to_each_band(self):
        y, scores = three_class()
        bands = roc_band_ovr(y, scores, grid_size=17, n_boot=1200, random_state=0)
        for band in bands.values():
            assert band.fpr.shape == (17,)
            assert band.n_boot == 1200

    def test_invalid_family_raises(self):
        y, scores = three_class()
        with pytest.raises(RocciError, match="family"):
            roc_band_ovr(
                y,
                scores,
                family="sidak",  # ty: ignore[invalid-argument-type]
                random_state=0,
            )

    def test_column_count_mismatch_raises(self):
        y, scores = three_class()
        with pytest.raises(RocciError, match="n, m=3"):
            roc_band_ovr(y, scores[:, :2], random_state=0)

    def test_two_classes_rejected(self):
        y = np.repeat([0, 1], 40)
        scores = np.random.default_rng(0).random((80, 2))
        with pytest.raises(RocciError, match="m > 2"):
            roc_band_ovr(y, scores, random_state=0)

    def test_class_absent_from_labels_raises(self):
        # a wrong `classes` entry must not surface as a cryptic one-class error
        y, scores = three_class()
        with pytest.raises(RocciError, match=r"do not occur in y_true"):
            roc_band_ovr(y, scores, classes=[0, 1, 9], random_state=0)

    def test_duplicate_classes_raise(self):
        y, scores = three_class()
        with pytest.raises(RocciError, match="duplicates"):
            roc_band_ovr(y, scores, classes=[0, 1, 1], random_state=0)


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


def test_ovr_is_bitwise_a_manual_bonferroni_loop():
    """The OvR reduction is exactly the documented manual loop, bit for bit.

    Rebuilds every ingredient independently from the spec §3.6 contract —
    one-vs-rest labels ``y == cls``, column ``j`` for ``classes[j]``,
    per-class confidence ``1 - alpha/m``, and per-class seeds via
    ``SeedSequence(random_state).spawn(m)`` — and demands identical band
    arrays. Comparing arrays (not the ``confidence`` field, which
    ``roc_band_ovr`` sets by assignment) means a wrong column routing, an
    uncorrected alpha actually used in the computation, or a drifted seed
    derivation each produce different bootstrap draws or widths and fail.
    """
    y, scores = three_class(seed=2)
    bands = roc_band_ovr(y, scores, confidence=0.95, random_state=3)
    seeds = np.random.SeedSequence(3).spawn(3)
    for j, cls in enumerate((0, 1, 2)):
        seed_j = int(seeds[j].generate_state(1, dtype=np.uint64)[0])
        manual = roc_band(
            y == cls, scores[:, j], confidence=1.0 - 0.05 / 3.0, random_state=seed_j
        )
        assert bands[cls].random_state == seed_j
        assert bands[cls].confidence == pytest.approx(manual.confidence)
        np.testing.assert_array_equal(bands[cls].lower, manual.lower)
        np.testing.assert_array_equal(bands[cls].upper, manual.upper)
        np.testing.assert_array_equal(bands[cls].tpr, manual.tpr)
        assert bands[cls].auc == manual.auc
