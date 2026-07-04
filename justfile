# Canonical dev commands — identical locally and in CI (CI calls these recipes).

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# uv sync + optimized in-place extension build + pre-commit install
setup:
    uv sync --all-groups
    uv run maturin develop --release
    uv run pre-commit install

# Fast suite; excludes slow/calibration
test:
    uv run pytest -x -q -m "not slow"

# Full matrix incl. slow + a pure-NumPy fallback pass
test-all: test-fallback
    uv run pytest -q

# Full fast suite forced onto the NumPy fallback backend
test-fallback $ROCCI_BACKEND="numpy":
    uv run pytest -q -m "not slow"

lint:
    uv run ruff check .
    uv run ruff format --check .
    cargo fmt --all --check
    cargo clippy --all-targets -- -D warnings

fix:
    uv run ruff check --fix .
    uv run ruff format .
    uv run ty check --fix python/rocci tests
    cargo fmt --all

typecheck:
    uv run ty check

rust-test:
    cargo test

# benchmarks/ perf gates, prints table vs thresholds
bench:
    uv run python benchmarks/run_benchmarks.py

# Live-preview the docs site (vignettes execute on first render)
docs:
    uv run mkdocs serve

# Build the docs site strictly — what CI runs; failures include vignette errors
docs-build:
    uv run mkdocs build --strict

# Regenerate golden masters (requires a studroc_paper checkout; prints provenance).
# Policy: fixtures change ONLY with a spec change and a PR explaining the
# statistical delta (spec §5.7). Pass the paper repo's python explicitly.
fixtures paper_python="../studroc_paper/.venv/Scripts/python.exe":
    {{ paper_python }} scripts/record_golden_masters.py --out tests/fixtures/golden

# Version bump + git-cliff changelog + absolute perf check (spec §14.5)
release-prep version:
    uv run python scripts/release_prep.py {{ version }}
    just bench
