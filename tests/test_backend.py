"""Backend selection (§8.3), validation, determinism, and env override.

Risk mitigated: silent misrouting (wrong kernel, ignored override), and
input contracts failing without an actionable message.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import rocci
from rocci._exceptions import RocciError
from rocci.backend import BACKEND, bootstrap_tpr_matrix
from rocci.band.grids import grid_k_indices, make_grid
from tests.conftest import binormal_scores


def run_python(code: str, **env_overrides) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **{k: str(v) for k, v in env_overrides.items()}}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


class TestSelection:
    def test_backend_is_a_known_value(self):
        assert BACKEND in ("rust", "numpy")

    def test_version_is_exposed(self):
        assert rocci.__version__

    def test_env_override_numpy(self):
        proc = run_python(
            "from rocci.backend import BACKEND; print(BACKEND)",
            ROCCI_BACKEND="numpy",
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "numpy"

    def test_env_override_invalid_value_raises(self):
        proc = run_python("import rocci.backend", ROCCI_BACKEND="cuda")
        assert proc.returncode != 0
        assert "ROCCI_BACKEND" in proc.stderr

    def test_explicit_numpy_override_does_not_warn(self):
        proc = run_python(
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import rocci.backend\n"
            "print(sum('Fallback' in str(x.category) for x in w))",
            ROCCI_BACKEND="numpy",
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "0"


class TestValidation:
    def _args(self):
        neg, pos = binormal_scores(10, 10, seed=0)
        neg, pos = np.sort(neg), np.sort(pos)
        k = grid_k_indices(make_grid(10), 10)
        return neg, pos, k

    def test_empty_class_raises_with_fix(self):
        neg, pos, k = self._args()
        with pytest.raises(RocciError, match="non-empty"):
            bootstrap_tpr_matrix(np.array([]), pos, k, 10, 0)

    def test_empty_grid_raises(self):
        neg, pos, _ = self._args()
        with pytest.raises(RocciError, match="grid"):
            bootstrap_tpr_matrix(neg, pos, np.array([], dtype=np.uint64), 10, 0)

    def test_n_boot_below_one_raises(self):
        neg, pos, k = self._args()
        with pytest.raises(RocciError, match="n_boot"):
            bootstrap_tpr_matrix(neg, pos, k, 0, 0)

    def test_out_of_range_k_raises(self):
        neg, pos, _ = self._args()
        bad = np.array([0, 11], dtype=np.uint64)
        with pytest.raises(RocciError, match=r"\[0, n_neg=10\]"):
            bootstrap_tpr_matrix(neg, pos, bad, 10, 0)


class TestDeterminism:
    def test_same_seed_same_backend_bit_identical(self):
        neg, pos = binormal_scores(50, 50, seed=1)
        neg, pos = np.sort(neg), np.sort(pos)
        k = grid_k_indices(make_grid(50), 50)
        a = bootstrap_tpr_matrix(neg, pos, k, 200, seed=42)
        b = bootstrap_tpr_matrix(neg, pos, k, 200, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        neg, pos = binormal_scores(50, 50, seed=2)
        neg, pos = np.sort(neg), np.sort(pos)
        k = grid_k_indices(make_grid(50), 50)
        a = bootstrap_tpr_matrix(neg, pos, k, 200, seed=1)
        b = bootstrap_tpr_matrix(neg, pos, k, 200, seed=2)
        assert not np.array_equal(a, b)
