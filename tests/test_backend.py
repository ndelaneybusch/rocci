"""Backend selection, validation, determinism, and env override.

These tests govern the router that sits in front of both kernels — they say
nothing about whether a kernel computes the right numbers (that is
``test_fallback_kernel.py`` and ``test_rust_backend.py``), only that the right
kernel is chosen, that its shared input contract is enforced identically, and
that a given (data, seed) reproduces.

Guaranteed. ``BACKEND`` resolves to one of ``{"rust", "numpy"}`` and
``ROCCI_BACKEND`` overrides it in a fresh interpreter, with an invalid value
failing at import time rather than silently falling through. The fallback path
warns exactly once, and only when the Rust core is genuinely absent — an
explicit ``numpy`` override or a present Rust core imports silently. The shared
validation layer rejects an empty class, an empty grid, ``n_boot < 1``, and
out-of-range grid indices with a ``RocciError`` that names the bound. Within one
backend, the same (neg, pos, k, n_boot, seed) is bit-identical across calls, and
different seeds produce different draws.

Limitations. Determinism is checked per backend in isolation; cross-backend
bit-equality is deliberately not claimed. The env-override tests spawn
subprocesses, so they assert selection at import, not hot-swapping within a live
process.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

import rocci
import rocci.backend as backend_mod
from rocci import FallbackBackendWarning
from rocci._exceptions import RocciError
from rocci.backend import BACKEND, _fallback, bootstrap_tpr_matrix
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
            "from rocci.backend import BACKEND; print(BACKEND)", ROCCI_BACKEND="numpy"
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "numpy"

    def test_env_override_invalid_value_raises(self):
        proc = run_python("import rocci.backend", ROCCI_BACKEND="cuda")
        assert proc.returncode != 0
        assert "ROCCI_BACKEND" in proc.stderr

    @pytest.mark.skipif(BACKEND != "rust", reason="requires the Rust core")
    def test_rust_import_does_not_warn(self):
        # FallbackBackendWarning is declared for exactly one circumstance:
        # the Rust core missing. With the core present the import is silent.
        proc = run_python(
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import rocci.backend\n"
            "print(sum('Fallback' in str(x.category) for x in w))"
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "0"

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
        _, pos, k = self._args()
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


class TestRoutingMocked:
    """Dispatch inside ``bootstrap_tpr_matrix``, exercised in-process by swapping
    the module-level kernel handle. No subprocess and no real backend switch, so
    both arms of the router are covered whichever kernel the wheel actually
    shipped — the arm that isn't installed is otherwise dead in coverage."""

    def _args(self):
        neg, pos = binormal_scores(10, 10, seed=0)
        neg, pos = np.sort(neg), np.sort(pos)
        k = grid_k_indices(make_grid(10), 10)
        return neg, pos, k

    def test_numpy_kernel_used_when_rust_absent(self, monkeypatch):
        # Clearing the Rust handle forces the NumPy branch; the result must be
        # exactly what that kernel computes on its own.
        neg, pos, k = self._args()
        monkeypatch.setattr(backend_mod, "_rust_kernel", None)
        got = bootstrap_tpr_matrix(neg, pos, k, 64, seed=0)
        expected = _fallback.bootstrap_tpr_matrix_numpy(neg, pos, k, 64, seed=0)
        np.testing.assert_array_equal(got, expected)

    def test_rust_kernel_is_dispatched_when_present(self, monkeypatch):
        # A spy in the Rust slot proves the router calls it and forwards the
        # thread count, on a NumPy-only machine too (where the real arm is dead).
        neg, pos, k = self._args()
        seen = {}

        def spy(neg_, pos_, k_, n_boot, seed, n_threads):
            seen["args"] = (n_boot, seed, n_threads)
            return np.zeros((n_boot, len(k_)))

        monkeypatch.setattr(backend_mod, "_rust_kernel", spy)
        out = bootstrap_tpr_matrix(neg, pos, k, 8, seed=3, n_threads=2)
        assert out.shape == (8, len(k))
        assert seen["args"] == (8, 3, 2)

    def test_descending_k_indices_raise_rocci_error(self):
        # The router rejects a non-ascending grid before either kernel runs; the
        # NumPy fallback would otherwise return silently wrong values.
        neg, pos, _ = self._args()
        descending = np.array([5, 3, 1], dtype=np.uint64)
        with pytest.raises(RocciError, match="ascending"):
            bootstrap_tpr_matrix(neg, pos, descending, 8, seed=0)


_UNSET = object()


@pytest.fixture
def reload_backend():
    """Reload ``rocci.backend`` under a mocked import environment, then restore
    the real module state so later tests see the true backend.

    Backend selection runs once at import, so the only in-process way to reach
    its branches — without the subprocess cost of the tests above — is to re-run
    the module body with ``ROCCI_BACKEND`` and the ``rocci._core`` slot in
    ``sys.modules`` swapped out. The finalizer owns the restore itself rather
    than relying on ``monkeypatch`` teardown ordering, which is not guaranteed to
    undo the environment before the cleanup reload runs.
    """
    import os

    orig_env = os.environ.get("ROCCI_BACKEND")
    orig_core_mod = sys.modules.get("rocci._core", _UNSET)
    # Touch the package namespace through __dict__: the compiled ``_core`` is
    # invisible to static analysis, so plain attribute access would not type.
    orig_core_attr = rocci.__dict__.get("_core", _UNSET)

    def _reload(*, env=None, break_core=False):
        if env is None:
            os.environ.pop("ROCCI_BACKEND", None)
        else:
            os.environ["ROCCI_BACKEND"] = env
        if break_core:
            # ``from rocci import _core`` resolves the already-bound package
            # attribute before it ever consults ``sys.modules``, so both must go
            # for the import to raise — the state of a wheel built without the
            # compiled core.
            # A None entry is the CPython idiom for "this import must fail"; ty
            # types sys.modules as dict[str, ModuleType] and cannot model it.
            sys.modules["rocci._core"] = None  # ty: ignore[invalid-assignment]
            rocci.__dict__.pop("_core", None)
        return importlib.reload(backend_mod)

    yield _reload

    if orig_env is None:
        os.environ.pop("ROCCI_BACKEND", None)
    else:
        os.environ["ROCCI_BACKEND"] = orig_env
    if orig_core_mod is _UNSET:
        sys.modules.pop("rocci._core", None)
    else:
        # orig_core_mod is a real module here (not the sentinel), but ty cannot
        # narrow through the identity check against an ``object()`` sentinel.
        sys.modules["rocci._core"] = orig_core_mod  # ty: ignore[invalid-assignment]
    if orig_core_attr is _UNSET:
        rocci.__dict__.pop("_core", None)
    else:
        rocci.__dict__["_core"] = orig_core_attr
    importlib.reload(backend_mod)


class TestSelectionMocked:
    """Import-time backend selection, exercised in-process via a module reload
    instead of the subprocesses in :class:`TestSelection` — the same branches at
    a fraction of the cost, and counted by coverage."""

    def test_numpy_override_selects_numpy(self, reload_backend):
        mod = reload_backend(env="numpy")
        assert mod.BACKEND == "numpy"
        assert mod._rust_kernel is None

    @pytest.mark.skipif(BACKEND != "rust", reason="requires the Rust core")
    def test_rust_override_selects_rust(self, reload_backend):
        mod = reload_backend(env="rust")
        assert mod.BACKEND == "rust"
        assert mod._rust_kernel is not None

    def test_invalid_override_raises_at_import(self, reload_backend):
        with pytest.raises(RocciError, match="ROCCI_BACKEND"):
            reload_backend(env="cuda")

    def test_auto_falls_back_when_core_missing(self, reload_backend):
        with pytest.warns(FallbackBackendWarning, match="pure-NumPy"):
            mod = reload_backend(env=None, break_core=True)
        assert mod.BACKEND == "numpy"
        assert mod._rust_kernel is None
