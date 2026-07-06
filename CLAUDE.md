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

## Statistical semantics are locked — read before changing statistics

The tie and edge semantics in `band/` and both kernels (`searchsorted` sides, strict-vs-non-strict comparisons, sentinel handling, endpoint pins) encode validated statistical behavior. Any edit that changes one of them is a statistical change, not a cleanup, no matter how much it looks like a simplification.

- **Golden-master fixtures** (`tests/fixtures/golden`, checked by `test_golden_master.py`) were recorded once from the validated paper implementation and are the ground truth for the band assembly. Precedence: if a golden test disagrees with the code, **the fixture wins** — never regenerate fixtures to match new code. Fixtures change only through a PR that explains the statistical delta and its coverage consequences (see CONTRIBUTING.md §Golden-master fixtures); `just fixtures` regenerates them from a `studroc_paper` checkout.
- The band assembly order (envelope → Wilson rectangle floor + monotonicity → Beta floor → pinned endpoints) in `band/envelope.py` is load-bearing; do not reorder.
- Reproducibility contract: same seed + same backend + same version → bit-identical output, independent of thread count. Rust and NumPy backends use different RNG streams — they agree statistically, not bit-wise.

## Architecture

Layers, from outside in:

- `python/rocci/__init__.py` — deliberately small public surface: `roc_band`, `roc_band_ovr`, `from_estimator`, `RocBand`, `show_versions`, plus the exception/warning taxonomy. Everything else is private (`rocci._*`) or documented internal.
- `python/rocci/_api.py` — orchestration only: ingestion, argument validation (`_validation.py`), the warning taxonomy (`_warnings.py`), result assembly. **No new statistics live here.**
- `python/rocci/ingest.py` — coerces NumPy/pandas/polars/torch/JAX/sequences to the `(neg, pos)` score split. All duck-typed/protocol-based: rocci never imports those libraries (numpy is the only runtime dep).
- `python/rocci/band/` — the statistical core. `grids.py` (FPR grid, empirical ROC step interpolation), `envelope.py` (studentized envelope, assembly, attribution, AUC), `floors.py` (Wilson variance floor, gated Wilson rectangle floor, Beta order-statistic floor), `normal.py` (Working–Hotelling band + normality diagnostics for `normal=True`).
- `python/rocci/backend/` — kernel routing. Uses the Rust core when importable, else the pure-NumPy fallback (`_fallback.py`) with a one-time `FallbackBackendWarning`. `ROCCI_BACKEND={rust,numpy}` overrides selection (rust raises if the extension is missing). Shared input validation lives in `backend/__init__.py` so error behavior is identical across backends.
- `rust/src/lib.rs` — the bootstrap TPR kernel (rayon-parallel, xoshiro256++ seeded per replicate). Internal crate version stays `0.0.0`; the single version source is `[project]` in `pyproject.toml`.
- `python/rocci/_result.py` — frozen dataclasses `RocBand` / `NormalityReport`; `plotting.py` is behind the optional `rocci[plot]` extra.

## Parallel work: one task = one worktree = one branch = one PR

Concurrent agents (or tasks) must never share a checkout — simultaneous edits to one working tree collide silently and contaminate each other's commits. Isolate every task:

- **Branch** from up-to-date `main`, named `<type>/<short-slug>` where `<type>` is the Conventional Commit type the eventual PR title will carry (`fix/ingest-bool-pos-label`, `test/floor-oracles`, `ci/gates-workflow`). Choosing the type up front forces the scope decision early: if you can't pick one type, the task is two branches.
- **Worktree** per branch, as a sibling directory: `git worktree add ../rocci-<slug> -b <type>/<slug> main`, then `just setup` inside it. Each worktree gets its own `.venv`, Rust `target/`, and compiled extension — nothing is shared except the git object store, so builds and test runs cannot interfere. Remove with `git worktree remove ../rocci-<slug>` after merge.
- **PR** per branch, squash-merged. The PR title becomes the sole commit on `main` and is CI-checked against `^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|release|revert)(\(.+\))?!?: .+` (lowercase type). git-cliff builds the changelog from these titles, so one PR = one type = one changelog entry — keep PRs single-facet, and peel a user-visible `fix` out of a `test`/`ci` sweep into its own small PR rather than letting the squash bury it.
- Keep branches hours-to-days short (trunk-based); if `main` moves, rebase the worktree branch rather than merging `main` into it — branch protection requires linear history.

## Conventions

- Trunk-based, squash merge; PR titles follow Conventional Commits (CI-enforced — the PR title becomes the commit message). `CHANGELOG.md` is git-cliff-generated at release; don't hand-edit.
- Google docstrings on every public object, each with a runnable `Examples:` block — doctests run in CI.
- Type checking is `ty` against the 3.10 floor, not the dev interpreter. `unused-ignore-comment` is an error: suppressions must stay honest.
- Rust: clippy pedantic with `-D warnings`; the allow-list is documented in `rust/Cargo.toml`. `unsafe` only at the FFI boundary with `// SAFETY:` comments.
- Comments are evergreen; no commented-out code; no TODOs without a linked issue number. DO provide context and insights in docs and comments that let an interested and informed reader understand the purpose and implementation quickly. DO NOT reference design documents, external documents, prior code states, alternative implementations that the current implementation does not use etc.
- Vignettes are jupytext `.md` files in `docs/vignettes/`, executed at docs build time (`just docs-build` catches breakage).


## Releasing ("let's do a new release")

The full operator guide is CONTRIBUTING.md §Releasing; the agent-shaped short
version:

1. Preflight: `main` clean and green; pick the SemVer version (`X.Y.Z`, or
   `X.Y.ZrcN` for a TestPyPI release candidate).
2. `just release-prep X.Y.Z` — sets the version, regenerates CHANGELOG.md,
   and runs the absolute perf gates locally. Review the diff.
3. Open a PR titled exactly `release: vX.Y.Z` from a `release/vX.Y.Z`
   branch. The owner reviews and merges (agents cannot merge their own PRs).
4. If wheels.yml changed since the last release, validate it first with the
   `workflow_dispatch` dry-run (`gh workflow run wheels.yml`) — it builds and
   smoke-tests everything but cannot publish. Cheaper than burning a tag.
5. Tag the merge commit — **annotated**, on `main`:
   `git tag -a vX.Y.Z -m "rocci X.Y.Z" && git push origin vX.Y.Z`.
6. Watch the run (`gh run watch`). It pauses at the `release` environment;
   only the owner can approve — tell them when it's waiting. Everything after
   approval (publish, post-publish verification on three OSes, GitHub
   Release/Zenodo, docs `stable` alias) is automatic.
7. A failure before approval publishes nothing: fix via PR, merge, then move
   the tag to the fixed commit (`git push origin :refs/tags/vX.Y.Z`, re-tag,
   push). Moving a remote tag is destructive — get the owner's explicit OK.

Owner-only, and required once per machine/account rather than per release:
PyPI/TestPyPI trusted publishers, `release` environment approval rights,
Zenodo webhook. There is deliberately no conda package (see
CONTRIBUTING.md §Releasing) — do not offer to create a feedstock unless the
owner raises it.
