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
      match new code; spec §5.7 precedence rule)
- [ ] If the spec changed, this PR links the spec delta and explains the
      statistical consequences
