# Installation

```bash
pip install rocci            # core: numpy + scipy only
pip install 'rocci[plot]'    # + matplotlib for band.plot() / diagnostics
```

Binary wheels ship for Linux (x86-64, aarch64, musl), macOS (Intel and Apple
silicon), and Windows, for every Python ≥ 3.10 — `pip install rocci` never
compiles Rust on a supported platform.

## conda-forge

Once the feedstock lands (after the first PyPI release):

```bash
conda install -c conda-forge rocci
```

## The fallback backend

rocci's bootstrap kernel is compiled Rust. On a platform with no wheel and no
Rust toolchain, the package still works: a pure-NumPy kernel with identical
statistical semantics takes over automatically, and rocci emits a
`FallbackBackendWarning` once per process so you know you are on the slow
path (10–30× slower — usually still well under a second). Which backend is
active is recorded on every result as `band.backend` and printed by
`rocci.show_versions()`.

There is nothing to configure: backend selection is automatic and does not
change any statistical output contract. Same seed + same backend + same
version ⇒ bit-identical bands; across backends, results agree statistically
but not bit-for-bit (different RNG streams — documented, tested).

## Requirements

| | |
|---|---|
| Python | ≥ 3.10 |
| Hard dependencies | `numpy >= 1.24`, `scipy >= 1.10` |
| Optional | `matplotlib >= 3.7` (`rocci[plot]`), `pandas` (only for `to_dataframe()`) |

## For bug reports

```python
import rocci
rocci.show_versions()
```

prints the version, active backend, dependency versions, OS, and CPU count —
paste it into the issue template.
