# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rocci` — distribution-free simultaneous confidence bands for ROC curves. A Python package (`python/rocci/`) with a Rust bootstrap kernel (`rust/`) built via maturin into `rocci._core`.

## Commands

The `justfile` is the single answer to "how do I run X" — CI calls the same recipes.

```
just setup          # uv sync --all-groups + maturin develop --release + pre-commit install
just test           # fast suite (excludes -m slow)
just test-all       # full matrix, incl. slow + a ROCCI_BACKEND=numpy pass
just rust-test      # cargo test
just lint           # ruff check/format --check + cargo fmt --check + clippy -D warnings
just fix            # apply ruff/ty/cargo autofixes
just typecheck      # ty
just bench          # benchmarks/ perf gates vs thresholds
just docs-build     # mkdocs build --strict (what CI runs; catches vignette errors)
```

Single test: `uv run pytest tests/test_envelope.py -k name_fragment`. Slow statistical suites (cross-backend agreement at B=8000, calibration) are marked `-m slow`.

After editing Rust, rebuild the extension with `uv run maturin develop --release` before running Python tests.

## Normative spec — read before changing statistics

`rocci_spec.md` and `rocci_spec_appendix.md` fully determine the implementation. Appendix routines are labeled A1–A16; ones marked **EXACT** encode validated tie/edge semantics and must be implemented as written.

- **Golden-master fixtures** (`tests/fixtures/golden`, checked by `test_golden_master.py`) were recorded once from the validated paper implementation. Precedence: if a golden test disagrees with an appendix routine, **the fixture wins** — never regenerate fixtures to match new code. Fixtures change only with a spec change and a PR explaining the statistical delta (spec §5.7); `just fixtures` regenerates them from a `studroc_paper` checkout.
- The band assembly order (envelope → Wilson rectangle floor + monotonicity → Beta floor → pinned endpoints) in `band/envelope.py` is load-bearing; do not reorder.
- Reproducibility contract: same seed + same backend + same version → bit-identical output, independent of thread count. Rust and NumPy backends use different RNG streams — they agree statistically, not bit-wise.

## Architecture

Layers, from outside in:

- `python/rocci/__init__.py` — deliberately small public surface: `roc_band`, `roc_band_ovr`, `from_estimator`, `RocBand`, `show_versions`, plus the exception/warning taxonomy. Everything else is private (`rocci._*`) or documented internal.
- `python/rocci/_api.py` — orchestration only: ingestion, argument validation (`_validation.py`), the warning taxonomy (`_warnings.py`), result assembly. **No new statistics live here.**
- `python/rocci/ingest.py` — coerces NumPy/pandas/polars/torch/JAX/sequences to the `(neg, pos)` score split. All duck-typed/protocol-based: rocci never imports those libraries (runtime deps are numpy + scipy only).
- `python/rocci/band/` — the statistical core. `grids.py` (FPR grid, empirical ROC step interpolation), `envelope.py` (studentized envelope, assembly, attribution, AUC), `floors.py` (Wilson variance floor, gated Wilson rectangle floor, Beta order-statistic floor), `normal.py` (Working–Hotelling band + normality diagnostics for `normal=True`).
- `python/rocci/backend/` — kernel routing. Uses the Rust core when importable, else the pure-NumPy fallback (`_fallback.py`) with a one-time `FallbackBackendWarning`. `ROCCI_BACKEND={rust,numpy}` overrides selection (rust raises if the extension is missing). Shared input validation lives in `backend/__init__.py` so error behavior is identical across backends.
- `rust/src/lib.rs` — the bootstrap TPR kernel (rayon-parallel, xoshiro256++ seeded per replicate). Internal crate version stays `0.0.0`; the single version source is `[project]` in `pyproject.toml`.
- `python/rocci/_result.py` — frozen dataclasses `RocBand` / `NormalityReport`; `plotting.py` is behind the optional `rocci[plot]` extra.

## Conventions

- Trunk-based, squash merge; PR titles follow Conventional Commits (CI-enforced — the PR title becomes the commit message). `CHANGELOG.md` is git-cliff-generated at release; don't hand-edit.
- Google docstrings on every public object, each with a runnable `Examples:` block — doctests run in CI.
- Type checking is `ty` against the 3.10 floor, not the dev interpreter. `unused-ignore-comment` is an error: suppressions must stay honest.
- Rust: clippy pedantic with `-D warnings`; the allow-list is documented in `rust/Cargo.toml`. `unsafe` only at the FFI boundary with `// SAFETY:` comments.
- Comments are evergreen; no commented-out code; no TODOs without a linked issue number. DO provide context and insights in docs and comments that let an interested and informed reader understand the purpose and implementation quickly. DO NOT reference design documents, external documents, prior code states, alternative implementations that the current implementation does not use etc.
- Vignettes are jupytext `.md` files in `docs/vignettes/`, executed at docs build time (`just docs-build` catches breakage).

