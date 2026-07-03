"""Performance gates measured with time.perf_counter.

"End-to-end" means the full statistical pipeline: split + sort + grid +
kernel + assembly.

Release gates (12-core desktop class, defaults n_boot=2000, K=512):
  n=10k  < 100 ms | n=100k < 500 ms | n=1M < 3 s
  non-bootstrap overhead at n=100k < 50 ms

Exit code 1 on any breach with --strict (used by release-prep; the per-PR
CI perf job measures relative-to-main instead).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from rocci.backend import BACKEND, bootstrap_tpr_matrix
from rocci.band.envelope import assemble_envelope_band
from rocci.band.grids import grid_k_indices, make_grid

GATES_S = {10_000: 0.100, 100_000: 0.500, 1_000_000: 3.000}
OVERHEAD_GATE_S = 0.050  # at n=100k
N_BOOT = 2000


def pipeline(neg, pos, n_boot=N_BOOT, seed=0):
    t0 = time.perf_counter()
    neg_s, pos_s = np.sort(neg), np.sort(pos)
    grid = make_grid(len(neg_s))
    k = grid_k_indices(grid, len(neg_s))
    t1 = time.perf_counter()
    boot = bootstrap_tpr_matrix(neg_s, pos_s, k, n_boot, seed)
    t2 = time.perf_counter()
    assemble_envelope_band(boot, grid, neg_s, pos_s, alpha=0.05)
    t3 = time.perf_counter()
    return t3 - t0, (t1 - t0) + (t3 - t2)  # total, non-bootstrap overhead


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit 1 on breach")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    print(f"backend: {BACKEND}  (gates assume rust)")
    print(f"{'n_total':>10} {'total_s':>9} {'overhead_s':>11} {'gate_s':>7}  verdict")
    breached = False
    for n_total, gate in GATES_S.items():
        half = n_total // 2
        neg = rng.normal(0, 1, half)
        pos = rng.normal(1, 1, half)
        best_total = min(pipeline(neg, pos)[0] for _ in range(args.repeats))
        best_overhead = min(pipeline(neg, pos)[1] for _ in range(args.repeats))
        ok = best_total < gate
        overhead_ok = best_overhead < OVERHEAD_GATE_S if n_total == 100_000 else True
        breached |= not (ok and overhead_ok)
        verdict = "PASS" if (ok and overhead_ok) else "FAIL"
        print(
            f"{n_total:>10} {best_total:>9.3f} {best_overhead:>11.3f} "
            f"{gate:>7.3f}  {verdict}"
        )
    if breached and BACKEND != "rust":
        print("note: gates are defined for the rust backend")
    return 1 if (breached and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
