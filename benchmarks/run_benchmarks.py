"""Performance gates measured with time.perf_counter.

"End-to-end" means the full statistical pipeline: split + sort + grid +
kernel + assembly.

Release gates (12-core desktop class, defaults n_boot=2000, K=512):
  n=10k  < 100 ms | n=100k < 500 ms | n=1M < 3 s
  non-bootstrap overhead at n=100k < 50 ms

Exit code 1 on any breach with --strict. Two modes:

- absolute (default): every timing must beat its release gate — used by
  `just bench` and release-prep on developer hardware.
- relative (--compare-json): used by the per-PR CI perf job to absorb
  runner variance. A timing fails on a regression past --relative-tolerance
  (default 30%) versus the baseline, or on breaching 2x the absolute release
  gate — the backstop that stops a slow-runner baseline from laundering a
  real regression.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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


def run_suite(repeats: int) -> dict[str, Any]:
    """Run the benchmark suite and return machine-readable results."""
    rng = np.random.default_rng(0)
    results = []
    for n_total, gate in GATES_S.items():
        half = n_total // 2
        neg = rng.normal(0, 1, half)
        pos = rng.normal(1, 1, half)
        totals = []
        overheads = []
        for _ in range(repeats):
            total_s, overhead_s = pipeline(neg, pos)
            totals.append(total_s)
            overheads.append(overhead_s)
        results.append(
            {
                "n_total": n_total,
                "total_s": min(totals),
                "overhead_s": min(overheads),
                "gate_s": gate,
                "overhead_gate_s": OVERHEAD_GATE_S if n_total == 100_000 else None,
            }
        )
    return {"backend": BACKEND, "n_boot": N_BOOT, "results": results}


def load_results(path: Path) -> dict[int, dict[str, Any]]:
    """Load a previous JSON benchmark result keyed by n_total."""
    payload = json.loads(path.read_text())
    return {int(row["n_total"]): row for row in payload["results"]}


def print_results(
    payload: dict[str, Any], baseline: dict[int, dict[str, Any]] | None
) -> bool:
    """Print benchmark results and return whether any gate was breached."""
    print(f"backend: {payload['backend']}  (gates assume rust)")
    if baseline is None:
        print(
            f"{'n_total':>10} {'total_s':>9} {'overhead_s':>11} {'gate_s':>7}  verdict"
        )
    else:
        print(
            f"{'n_total':>10} {'total_s':>9} {'base_s':>9} "
            f"{'overhead_s':>11} {'base_ovh':>9}  verdict"
        )

    breached = False
    for row in payload["results"]:
        n_total = int(row["n_total"])
        total_s = float(row["total_s"])
        overhead_s = float(row["overhead_s"])
        if baseline is None:
            ok = total_s < float(row["gate_s"])
            overhead_ok = overhead_s < OVERHEAD_GATE_S if n_total == 100_000 else True
            breached |= not (ok and overhead_ok)
            verdict = "PASS" if (ok and overhead_ok) else "FAIL"
            print(
                f"{n_total:>10} {total_s:>9.3f} {overhead_s:>11.3f} "
                f"{float(row['gate_s']):>7.3f}  {verdict}"
            )
        else:
            base = baseline[n_total]
            base_total = float(base["total_s"])
            base_overhead = float(base["overhead_s"])
            print(
                f"{n_total:>10} {total_s:>9.3f} {base_total:>9.3f} "
                f"{overhead_s:>11.3f} {base_overhead:>9.3f}  pending"
            )
    return breached


def compare_relative(
    payload: dict[str, Any], baseline: dict[int, dict[str, Any]], tolerance: float
) -> bool:
    """Return True when the current suite regresses past the baseline or backstop.

    A timing breaches on either criterion: slower than baseline by more than
    ``tolerance``, or above 2x its absolute release gate (the backstop
    described in the module docstring).
    """
    breached = False
    print(f"\nrelative tolerance: {tolerance:.0%}; absolute backstop: 2x release gates")
    for row in payload["results"]:
        n_total = int(row["n_total"])
        base = baseline[n_total]
        total_s = float(row["total_s"])
        total_ok = total_s <= float(base["total_s"]) * (
            1.0 + tolerance
        ) and total_s <= 2.0 * float(row["gate_s"])
        overhead_ok = True
        if n_total == 100_000:
            overhead_s = float(row["overhead_s"])
            overhead_ok = (
                overhead_s <= float(base["overhead_s"]) * (1.0 + tolerance)
                and overhead_s <= 2.0 * OVERHEAD_GATE_S
            )
        breached |= not (total_ok and overhead_ok)
        verdict = "PASS" if (total_ok and overhead_ok) else "FAIL"
        print(f"{n_total:>10}: {verdict}")
    return breached


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit 1 on breach")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, help="write benchmark results to JSON")
    parser.add_argument("--from-json", type=Path, help="read current results from JSON")
    parser.add_argument(
        "--compare-json",
        type=Path,
        help="compare against a baseline JSON result instead of absolute gates",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.30,
        help="allowed relative slowdown when --compare-json is used",
    )
    args = parser.parse_args()

    payload = (
        json.loads(args.from_json.read_text())
        if args.from_json is not None
        else run_suite(args.repeats)
    )
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")

    baseline = load_results(args.compare_json) if args.compare_json else None
    breached = print_results(payload, baseline)
    if baseline is not None:
        breached = compare_relative(payload, baseline, args.relative_tolerance)

    if breached and BACKEND != "rust":
        print("note: gates are defined for the rust backend")
    return 1 if (breached and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
