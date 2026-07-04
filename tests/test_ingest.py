"""Ingestion matrix.

Risk mitigated: rocci promises to accept whatever container users have with
zero hard dependencies, then split it into clean ``(neg, pos)`` scores. A
regression here silently corrupts every downstream band, so this suite red-teams
coercion, label resolution, score reduction, NaN handling, and the ties/small-
sample warnings across the containers and edge cases.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rocci._exceptions import RocciError
from rocci._warnings import SmallSampleWarning, TiesWarning
from rocci.ingest import ingest
from tests.conftest import binormal_scores


def labeled(n_neg=40, n_pos=40, seed=0, tie_step=None):
    """Return flat ``(y_true, y_score)`` with 0/1 labels for ingestion tests."""
    neg, pos = binormal_scores(n_neg, n_pos, seed=seed, tie_step=tie_step)
    y_true = np.concatenate([np.zeros(n_neg, int), np.ones(n_pos, int)])
    y_score = np.concatenate([neg, pos])
    return y_true, y_score


class _DlpackFails:
    """Torch-like: __dlpack__ raises (CUDA), falls back to detach().cpu().numpy()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def __dlpack__(self, *args, **kwargs):
        raise BufferError("tensor is on a CUDA device")

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _ArrayProto:
    """pandas/xarray-like object exposing __array__."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)


class _ToNumpy:
    """polars-like object exposing only .to_numpy()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def to_numpy(self):
        return self._arr


class TestContainerCoercion:
    @pytest.mark.parametrize(
        "wrap",
        [list, np.asarray, _ArrayProto, _ToNumpy, _DlpackFails],
        ids=["list", "ndarray", "array_proto", "to_numpy", "dlpack_fallback"],
    )
    def test_containers_split_identically(self, wrap):
        y_true, y_score = labeled()
        data = ingest(wrap(y_true), wrap(y_score))
        assert data.n_neg == 40
        assert data.n_pos == 40

    def test_non_numeric_scores_raise(self):
        with pytest.raises(RocciError, match="numeric"):
            ingest(np.array([0, 1, 0, 1]), np.array(["a", "b", "c", "d"]))

    def test_dlpack_non_cpu_without_transfer_raises(self):
        class OnlyDlpack:
            def __dlpack__(self, *a, **k):
                raise BufferError("cuda")

        with pytest.raises(RocciError, match="CPU"):
            ingest(np.array([0, 1, 0, 1]), OnlyDlpack())


class TestLabels:
    @pytest.mark.parametrize(
        ("labels", "expected_pos"),
        [
            (np.array([False, False, True, True]), 2),
            (np.array([0, 0, 1, 1]), 2),
            (np.array([-1, -1, 1, 1]), 2),
        ],
        ids=["bool", "zero_one", "neg_one_one"],
    )
    def test_positive_class_inferred(self, labels, expected_pos):
        data = ingest(labels, np.array([0.1, 0.2, 0.8, 0.9]))
        assert data.n_pos == expected_pos

    def test_string_labels_need_pos_label(self):
        y = np.array(["dog", "dog", "cat", "cat"])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        with pytest.raises(RocciError, match="pos_label"):
            ingest(y, s)
        data = ingest(y, s, pos_label="cat")
        assert data.n_pos == 2

    def test_bad_pos_label_raises(self):
        with pytest.raises(RocciError, match="not among"):
            ingest(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), pos_label=7)

    def test_more_than_two_labels_points_to_ovr(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        with pytest.raises(RocciError, match="roc_band_ovr"):
            ingest(y, np.arange(6.0))

    def test_single_class_raises(self):
        with pytest.raises(RocciError, match="one class"):
            ingest(np.ones(6, int), np.arange(6.0))

    def test_tiny_class_raises(self):
        with pytest.raises(RocciError, match="at least 2"):
            ingest(np.array([0, 0, 0, 1]), np.arange(4.0))

    def test_small_sample_warns_but_proceeds(self):
        y_true, y_score = labeled(n_neg=10, n_pos=10)
        with pytest.warns(SmallSampleWarning):
            data = ingest(y_true, y_score)
        assert data.n_neg == 10

    def test_small_sample_boundary_is_twenty(self):
        # declared threshold: warn when a class has fewer than 20 samples
        with warnings.catch_warnings():
            warnings.simplefilter("error", SmallSampleWarning)
            ingest(*labeled(n_neg=20, n_pos=20))
        with pytest.warns(SmallSampleWarning):
            ingest(*labeled(n_neg=19, n_pos=20))

    def test_bool_labels_honor_pos_label_false(self):
        # an explicit pos_label must never be silently overridden, even for
        # booleans: pos_label=False flips which class is positive
        y = np.array([False, False, False, True, True])
        s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        data = ingest(y, s, pos_label=False)
        assert data.n_pos == 3
        np.testing.assert_array_equal(np.sort(data.pos), [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(np.sort(data.neg), [0.8, 0.9])

    def test_bool_labels_pos_label_true_matches_inference(self):
        y = np.array([False, False, True, True])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        inferred = ingest(y, s)
        explicit = ingest(y, s, pos_label=True)
        np.testing.assert_array_equal(np.sort(inferred.pos), np.sort(explicit.pos))
        assert inferred.n_pos == explicit.n_pos == 2

    def test_all_true_bool_labels_raise_single_class(self):
        with pytest.raises(RocciError, match="one class"):
            ingest(np.ones(6, bool), np.arange(6.0))

    def test_nan_labels_raise_by_default(self):
        y = np.array([0.0, 0.0, np.nan, 1.0, 1.0])
        s = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        with pytest.raises(RocciError, match="NaN"):
            ingest(y, s)

    def test_nan_labels_omit_drops_row_not_class_count(self):
        # a NaN label must neither crash label resolution nor become a third
        # class; under omit exactly that row disappears
        y = np.array([0.0, 0.0, np.nan, 1.0, 1.0, 0.0, 1.0])
        s = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 0.15, 0.85])
        with pytest.warns(SmallSampleWarning, match="dropped 1"):
            data = ingest(y, s, nan_policy="omit")
        assert data.n_neg == 3
        assert data.n_pos == 3
        assert 0.5 not in np.concatenate([data.neg, data.pos])

    def test_omit_that_erases_a_class_raises(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, np.nan, np.nan])
        with (
            pytest.warns(SmallSampleWarning, match="dropped 2"),
            pytest.raises(RocciError, match="one class"),
        ):
            ingest(y, s, nan_policy="omit")

    def test_invalid_nan_policy_raises(self):
        y_true, y_score = labeled()
        with pytest.raises(RocciError, match="nan_policy"):
            ingest(y_true, y_score, nan_policy="drop")


class TestColumnVectors:
    def test_column_vector_labels_ravel(self):
        # sklearn/keras habit: y.reshape(-1, 1)
        y_true, y_score = labeled()
        data = ingest(y_true.reshape(-1, 1), y_score)
        assert data.n_neg == 40
        assert data.n_pos == 40

    def test_column_vector_scores_ravel(self):
        # predict_proba(X)[:, [1]] keeps a trailing singleton axis
        y_true, y_score = labeled()
        data = ingest(y_true, y_score.reshape(-1, 1))
        np.testing.assert_array_equal(np.sort(data.pos), np.sort(y_score[y_true == 1]))

    def test_higher_dim_labels_raise(self):
        y_true, y_score = labeled()
        with pytest.raises(RocciError, match="1-D"):
            ingest(np.tile(y_true, (2, 1)), y_score)

    def test_empty_labels_raise(self):
        with pytest.raises(RocciError, match="empty"):
            ingest(np.array([]), np.array([]))


class TestScores:
    def test_proba_matrix_takes_positive_column(self):
        y_true, raw = labeled()
        p1 = 1.0 / (1.0 + np.exp(-raw))
        proba = np.column_stack([1.0 - p1, p1])
        data = ingest(y_true, proba)
        assert data.notes  # an INFO note, not a warning
        np.testing.assert_allclose(np.sort(data.pos), np.sort(p1[y_true == 1]))

    def test_multiclass_proba_matrix_points_to_ovr(self):
        # a 3-column predict_proba must not surface as a "posterior draws" error
        y_true, _ = labeled()
        proba = np.random.default_rng(0).dirichlet(np.ones(3), size=len(y_true))
        with pytest.raises(RocciError, match="roc_band_ovr"):
            ingest(y_true, proba)

    def test_posterior_draws_require_score_reduce(self):
        y_true, raw = labeled()
        draws = np.tile(raw, (30, 1))  # (draws, n)
        with pytest.raises(RocciError, match="score_reduce"):
            ingest(y_true, draws)
        data = ingest(y_true, draws, score_reduce="mean")
        assert data.n_neg + data.n_pos == len(y_true)

    def test_three_dim_draws_reduce_over_draw_axes(self):
        y_true, raw = labeled()
        draws = np.tile(raw, (2, 5, 1))  # (chain, draw, n)
        data = ingest(y_true, draws, score_reduce="median")
        assert data.n_neg + data.n_pos == len(y_true)

    def test_length_mismatch_raises(self):
        with pytest.raises(RocciError, match="same samples"):
            ingest(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3]))

    def test_proba_matrix_honors_pos_label_column(self):
        # pos_label=0 selects column 0 of a predict_proba matrix, mirroring
        # sklearn's column ordering by sorted class label
        y_true, raw = labeled()
        p1 = 1.0 / (1.0 + np.exp(-raw))
        proba = np.column_stack([1.0 - p1, p1])
        data = ingest(y_true, proba, pos_label=0)
        # class 0 is now positive, scored by column 0 = 1 - p1
        np.testing.assert_allclose(np.sort(data.pos), np.sort(1.0 - p1[y_true == 0]))

    def test_two_column_non_probabilities_demand_score_reduce(self):
        # an (n, 2) matrix whose rows do not sum to 1 (e.g. paired logits)
        # must not be silently treated as probabilities
        y_true, raw = labeled()
        logits = np.column_stack([raw, -raw])
        with pytest.raises(RocciError, match="score_reduce"):
            ingest(y_true, logits)

    def test_invalid_score_reduce_raises(self):
        y_true, raw = labeled()
        draws = np.tile(raw, (5, 1))
        with pytest.raises(RocciError, match="score_reduce must be"):
            ingest(y_true, draws, score_reduce="max")

    def test_four_dimensional_scores_rejected(self):
        y_true, raw = labeled()
        with pytest.raises(RocciError, match="unsupported shape"):
            ingest(y_true, np.tile(raw, (2, 2, 2, 1)))

    def test_score_reduce_mean_is_the_column_mean(self):
        # the reduction must be over draw axes only, sample-aligned
        y_true, raw = labeled()
        rng = np.random.default_rng(1)
        draws = raw + rng.normal(0, 0.1, (30, len(raw)))
        data = ingest(y_true, draws, score_reduce="mean")
        expected = draws.mean(axis=0)
        got = np.sort(np.concatenate([data.neg, data.pos]))
        np.testing.assert_allclose(got, np.sort(expected), atol=0)


class TestNanAndInf:
    def test_nan_raises_by_default(self):
        y_true, y_score = labeled()
        y_score[0] = np.nan
        with pytest.raises(RocciError, match="NaN"):
            ingest(y_true, y_score)

    def test_nan_omit_drops_and_warns(self):
        y_true, y_score = labeled()
        y_score[0] = np.nan
        with pytest.warns(SmallSampleWarning, match="dropped"):
            data = ingest(y_true, y_score, nan_policy="omit")
        assert data.n_neg + data.n_pos == len(y_true) - 1

    def test_infinite_scores_are_legal(self):
        y_true, y_score = labeled()
        y_score[0] = np.inf
        y_score[-1] = -np.inf
        data = ingest(y_true, y_score)
        assert np.isinf(data.neg).any() or np.isinf(data.pos).any()


class TestTies:
    def test_heavy_ties_warn(self):
        y_true, y_score = labeled(tie_step=0.5)
        with pytest.warns(TiesWarning):
            ingest(y_true, y_score)

    def test_constant_class_proceeds_with_warning(self):
        y_true, y_score = labeled()
        y_score[y_true == 0] = 3.0  # negatives are constant
        with pytest.warns(TiesWarning, match="constant"):
            data = ingest(y_true, y_score)
        assert data.n_neg == 40

    def test_continuous_scores_do_not_warn(self):
        y_true, y_score = labeled()
        with warnings.catch_warnings():
            warnings.simplefilter("error", TiesWarning)
            ingest(y_true, y_score)

    def test_ties_boundary_is_half_distinct(self):
        # the declared trigger is distinct fraction strictly below 0.5:
        # exactly half distinct stays silent, one fewer distinct value warns
        y = np.concatenate([np.zeros(20, int), np.ones(20, int)])
        neg = np.arange(20, dtype=float)
        silent = np.concatenate([neg, neg])  # 20 distinct of 40 = exactly 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error", TiesWarning)
            ingest(y, silent)
        neg_tied = neg.copy()
        neg_tied[-1] = neg_tied[-2]  # 19 distinct of 40 < 0.5
        with pytest.warns(TiesWarning, match="19 distinct of 40"):
            ingest(y, np.concatenate([neg_tied, neg_tied]))
