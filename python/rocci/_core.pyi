"""Type stub for the compiled Rust core (rust/src/lib.rs, spec §8.1)."""

import numpy as np
from numpy.typing import NDArray

def bootstrap_tpr_matrix(
    neg_sorted: NDArray[np.float64],
    pos_sorted: NDArray[np.float64],
    k_indices: NDArray[np.uint64],
    n_boot: int,
    seed: int,
    n_threads: int,
) -> NDArray[np.float64]: ...
