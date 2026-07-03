"""Record golden-master fixtures using the reference studroc_paper implementation.

Run with the paper repo's environment, never rocci's:

    ../studroc_paper/.venv/Scripts/python.exe scripts/record_golden_masters.py \
        --out tests/fixtures/golden

Records, per cell: the inputs (boot_tpr_matrix, fpr_grid, y_true, y_score,
alpha) and the reference implementation's outputs for three arms—
"envelope" (full method), "envelope_no_beta_floor" (no beta floor), and
"envelope_pre_floor" (pre-floor only). rocci's assembly must reproduce all
of them within atol=1e-6.

Design notes (why these grids):
- The reference implementation computes in float32; rocci computes in float64.
  Step-function lookups (empirical ROC at grid points) are only reproducible
  across precisions if no grid point straddles an ROC vertex between the two
  precisions. Grid sizes are chosen so K-1 is a power of two (grid points
  are dyadic, hence exact in both float32 and float64, and every
  grid/vertex collision point is itself dyadic), except the n=10k cell
  where K=512 and gcd(511, 5000)=1 keeps grid points provably clear of
  vertices. Scores are drawn in float64 and cast to float32 once; the
  float32 values are canonical.
- CPU is forced for reproducibility of the recording.
"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import scipy
import torch
from scipy.stats import norm
from studroc_paper.methods.envelope_boot import (
    envelope_band_suite,
    envelope_bootstrap_band,
)
from studroc_paper.sampling.bootstrap_grid import generate_bootstrap_grid

ALPHA = 0.05
N_BOOT = 2000

# name -> (n_neg, n_pos, auc, grid_size, tie_step, seed)
CELLS = {
    "small_n_30_30": (30, 30, 0.80, 33, None, 101),
    "unbalanced_50_500": (50, 500, 0.80, 65, None, 202),
    "heavy_ties_200_200": (200, 200, 0.80, 129, 0.1, 303),
    "high_auc_300_300": (300, 300, 0.95, 257, None, 404),
    "n10k_5000_5000": (5000, 5000, 0.80, 512, None, 505),
}


def make_cell(n_neg, n_pos, auc, grid_size, tie_step, seed):
    d = float(np.sqrt(2.0) * norm.ppf(auc))
    rng = np.random.default_rng(seed)
    neg = rng.normal(0.0, 1.0, n_neg)
    pos = rng.normal(d, 1.0, n_pos)
    if tie_step is not None:
        neg = np.round(neg / tie_step) * tie_step
        pos = np.round(pos / tie_step) * tie_step
    y_true = np.concatenate([np.zeros(n_neg, np.int8), np.ones(n_pos, np.int8)])
    y_score = np.concatenate([neg, pos]).astype(np.float32)  # canonical values
    fpr_grid = np.linspace(0.0, 1.0, grid_size)  # float64, dyadic by design
    return y_true, y_score, fpr_grid


def record(name, spec, out_dir):
    n_neg, n_pos, auc, grid_size, tie_step, seed = spec
    y_true, y_score, fpr_grid = make_cell(*spec)

    torch.manual_seed(seed)
    boot = generate_bootstrap_grid(
        y_true=torch.from_numpy(y_true.astype(np.int64)),
        y_score=torch.from_numpy(y_score),
        B=N_BOOT,
        grid=torch.from_numpy(fpr_grid.astype(np.float32)),
        device=torch.device("cpu"),
    )
    boot_tpr = boot.numpy().astype(np.float32)

    _, lower, upper = envelope_bootstrap_band(
        boot_tpr_matrix=boot_tpr,
        fpr_grid=fpr_grid.astype(np.float32),
        y_true=y_true.astype(np.int64),
        y_score=y_score,
        alpha=ALPHA,
        boundary_method="wilson",
        retention_method="ks",
    )

    suite = envelope_band_suite(
        boot_tpr_matrix=boot_tpr,
        fpr_grid=fpr_grid.astype(np.float32),
        y_true=y_true.astype(np.int64),
        y_score=y_score,
        alphas=[ALPHA],
        include_pre_floor_arm=True,
    )[ALPHA]
    s_lower, s_upper = suite["envelope"]
    np.testing.assert_allclose(s_lower, lower, atol=1e-7)
    np.testing.assert_allclose(s_upper, upper, atol=1e-7)
    nb_lower, nb_upper = suite["envelope_no_beta_floor"]
    pf_lower, pf_upper = suite["envelope_pre_floor"]

    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        y_true=y_true,
        y_score=y_score,
        fpr_grid=fpr_grid,
        boot_tpr=boot_tpr,
        alpha=np.float64(ALPHA),
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        no_beta_lower=nb_lower.astype(np.float32),
        no_beta_upper=nb_upper.astype(np.float32),
        pre_floor_lower=pf_lower.astype(np.float32),
        pre_floor_upper=pf_upper.astype(np.float32),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(
        f"{name}: n=({n_neg},{n_pos}) auc={auc} K={grid_size} "
        f"ties={tie_step} seed={seed} -> {path.name} sha256:{digest}"
    )
    return name, spec, digest


def paper_git_sha():
    import studroc_paper

    repo = Path(studroc_paper.__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [record(name, spec, args.out) for name, spec in CELLS.items()]

    prov = args.out / "PROVENANCE.md"
    lines = [
        "# Golden-master provenance",
        "",
        "Recorded outputs of the reference `studroc_paper` implementation",
        "(`envelope_bootstrap_band(boundary_method='wilson')` and",
        "`envelope_band_suite`). These fixtures are the arbiters of the",
        "equivalence test. **Never regenerate them to match new code.**",
        "",
        f"- Recorded: {date.today().isoformat()}",
        "- Recorder: `scripts/record_golden_masters.py` (rocci repo)",
        f"- studroc_paper commit: `{paper_git_sha()}`",
        f"- Environment: python {platform.python_version()}, "
        f"torch {torch.__version__}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, {platform.system()} (CPU forced)",
        f"- alpha={ALPHA}, n_boot={N_BOOT}",
        "",
        "| cell | n_neg | n_pos | AUC | K | tie_step | seed | sha256[:16] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, (n0, n1, auc, k, ties, seed), digest in rows:
        lines.append(
            f"| {name} | {n0} | {n1} | {auc} | {k} | {ties} | {seed} | `{digest}` |"
        )
    lines += [
        "",
        "Grid sizes use K-1 = power of two (dyadic grid points) except the",
        "n=10k cell (K=512, gcd(511, 5000)=1) so float32/float64 step-lookup",
        "semantics provably agree; see the recorder's module docstring.",
    ]
    prov.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"provenance -> {prov}")


if __name__ == "__main__":
    sys.exit(main())
