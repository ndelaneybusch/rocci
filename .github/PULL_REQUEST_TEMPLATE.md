## What

<!-- One-paragraph summary. PR title must follow Conventional Commits
     (squash merge makes the title the commit message). -->

## Checklist

- [ ] `just test` and `just lint` pass locally
- [ ] New/changed behavior is covered by tests that mitigate a named risk
- [ ] Public objects have Google-style docstrings with runnable `Examples:`

### If this PR touches `python/rocci/band/`, `rust/`, or `tests/fixtures/golden/`

Statistical core — extra requirements:

- [ ] Golden-master tests pass **unchanged** (fixtures are never regenerated to
      match new code — if code and fixture disagree, the fixture wins)
- [ ] Any deliberate statistical change is explained in the PR description:
      what changed, and why coverage is preserved
