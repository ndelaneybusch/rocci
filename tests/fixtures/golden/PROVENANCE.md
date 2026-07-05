# Golden-master provenance

Recorded outputs of the validated `studroc_paper` implementation
(`envelope_bootstrap_band(boundary_method='wilson')` and
`envelope_band_suite`). These fixtures are the arbiters of the
golden-master equivalence suite (`tests/test_golden_master.py`).
**Never regenerate them to match new code.**

- Recorded: 2026-07-03
- Recorder: `scripts/record_golden_masters.py` (rocci repo)
- studroc_paper commit: `2eaac0a135487de234b54a91903b2dfd9ed40d06`
- Environment: python 3.12.11, torch 2.9.1+cu128, numpy 2.3.5, scipy 1.16.3, Windows (CPU forced)
- alpha=0.05, n_boot=2000

| cell | n_neg | n_pos | AUC | K | tie_step | seed | sha256[:16] |
|---|---|---|---|---|---|---|---|
| small_n_30_30 | 30 | 30 | 0.8 | 33 | None | 101 | `12e7549622e7033b` |
| unbalanced_50_500 | 50 | 500 | 0.8 | 65 | None | 202 | `ce61d217e450440c` |
| heavy_ties_200_200 | 200 | 200 | 0.8 | 129 | 0.1 | 303 | `d335d67ee89aff23` |
| high_auc_300_300 | 300 | 300 | 0.95 | 257 | None | 404 | `aace81647918c63a` |
| n10k_5000_5000 | 5000 | 5000 | 0.8 | 512 | None | 505 | `67c134ec16fbb36f` |

Grid sizes use K-1 = power of two (dyadic grid points) except the
n=10k cell (K=512, gcd(511, 5000)=1) so float32/float64 step-lookup
semantics provably agree; see the recorder's module docstring.
