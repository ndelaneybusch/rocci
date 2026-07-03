"""Bootstrap kernel backend selection (spec §8.3).

The Rust core (``rocci._core``) is used when importable; otherwise the
pure-NumPy fallback keeps the package functional everywhere, with a
:class:`~rocci._warnings.FallbackBackendWarning` emitted once per process.

``ROCCI_BACKEND={rust,numpy}`` overrides selection for testing/debugging
only (documented in CONTRIBUTING, not in user docs). There is no other
routing — performance is invisible to users.
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rocci._exceptions import RocciError
from rocci._warnings import FallbackBackendWarning
from rocci.backend import _fallback

__all__ = ["BACKEND", "bootstrap_tpr_matrix"]

BACKEND: Literal["rust", "numpy"]

_env = os.environ.get("ROCCI_BACKEND")
if _env == "numpy":
    # Explicit override: intentional, so no FallbackBackendWarning.
    _core = None
    BACKEND = "numpy"
elif _env == "rust":
    from rocci import _core  # raises ImportError loudly if the core is missing

    BACKEND = "rust"
elif _env is not None:
    raise RocciError(
        f"invalid ROCCI_BACKEND={_env!r}: set it to 'rust' or 'numpy', "
        "or unset it for automatic selection"
    )
else:
    try:
        from rocci import _core  # Rust

        BACKEND = "rust"
    except ImportError:
        _core = None
        BACKEND = "numpy"
        warnings.warn(
            "rocci's compiled Rust core is unavailable; using the pure-NumPy "
            "fallback kernel (10-30x slower, statistically identical). "
            "Install a binary wheel ('pip install rocci' on a supported "
            "platform) to get the fast backend.",
            FallbackBackendWarning,
            stacklevel=2,
        )


def bootstrap_tpr_matrix(
    neg_sorted: NDArray[np.float64],
    pos_sorted: NDArray[np.float64],
    k_indices: NDArray[np.uint64],
    n_boot: int,
    seed: int,
    n_threads: int | None = None,
) -> NDArray[np.float64]:
    """Compute the bootstrap TPR matrix on the active backend.

    Same seed + same backend + same version -> bit-identical output
    (independent of ``n_threads`` on the Rust backend). Rust and NumPy
    backends produce different RNG streams and agree statistically, not
    bit-wise (spec §8.4).

    Args:
        neg_sorted: Negative scores, ascending, finite-or-inf float64.
        pos_sorted: Positive scores, ascending.
        k_indices: Ascending order-statistic indices in ``[0, n_neg]``
            from :func:`rocci.band.grids.grid_k_indices`.
        n_boot: Number of bootstrap replicates (``>= 1``).
        seed: RNG seed, a non-negative integer below 2**64.
        n_threads: Rust thread count; ``None`` or 0 uses all cores.
            Ignored by the NumPy backend.

    Returns:
        Float64 array of shape ``(n_boot, len(k_indices))``.

    Raises:
        RocciError: On empty classes, an empty grid, ``n_boot < 1``, or
            out-of-range ``k_indices``.

    Examples:
        >>> import numpy as np
        >>> from rocci.backend import bootstrap_tpr_matrix
        >>> neg = np.linspace(-1, 1, 20)
        >>> pos = np.linspace(0, 2, 20)
        >>> k = np.array([0, 5, 10, 20], dtype=np.uint64)
        >>> out = bootstrap_tpr_matrix(neg, pos, k, n_boot=16, seed=3)
        >>> out.shape
        (16, 4)
        >>> reference = bootstrap_tpr_matrix(neg, pos, k, n_boot=16, seed=3)
        >>> bool((out == reference).all())  # same seed, same backend
        True
    """
    neg_sorted = np.ascontiguousarray(neg_sorted, dtype=np.float64)
    pos_sorted = np.ascontiguousarray(pos_sorted, dtype=np.float64)
    k_indices = np.ascontiguousarray(k_indices, dtype=np.uint64)

    n_neg = len(neg_sorted)
    if n_neg == 0 or len(pos_sorted) == 0:
        raise RocciError(
            "both classes must be non-empty to bootstrap; got "
            f"n_neg={n_neg}, n_pos={len(pos_sorted)}"
        )
    if len(k_indices) == 0:
        raise RocciError("k_indices is empty: the FPR grid needs at least 1 point")
    if n_boot < 1:
        raise RocciError(f"n_boot must be >= 1, got {n_boot}")
    if int(k_indices.max()) > n_neg:
        raise RocciError(
            f"k_indices values must lie in [0, n_neg={n_neg}]; "
            f"got max {int(k_indices.max())}"
        )

    if BACKEND == "rust":
        return _core.bootstrap_tpr_matrix(  # ty: ignore[possibly-unbound-attribute]
            neg_sorted,
            pos_sorted,
            k_indices,
            n_boot,
            seed,
            0 if n_threads is None else n_threads,
        )
    return _fallback.bootstrap_tpr_matrix_numpy(
        neg_sorted, pos_sorted, k_indices, n_boot, seed
    )
