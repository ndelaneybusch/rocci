# Installation

```bash
pip install rocci            # core: numpy + scipy only
pip install 'rocci[plot]'    # + matplotlib for band.plot() / diagnostics
```

Binary wheels ship for Linux (x86-64, aarch64, musl), macOS (Intel and Apple
silicon), and Windows, for every Python ≥ 3.10 — `pip install rocci` never
compiles Rust on a supported platform. Every wheel is smoke-tested on its own
platform in a clean environment before it can be published, and each release
is re-verified from the live index on all three OSes minutes after it goes
up.

## By tool

=== "pip"

    ```bash
    pip install 'rocci[plot]'
    pip install rocci==0.1.0        # pin for reproducible environments
    ```

=== "uv"

    ```bash
    uv add rocci                    # add to a uv-managed project
    uv pip install 'rocci[plot]'    # or into the active environment
    ```

=== "conda"

    ```bash
    conda activate myenv
    pip install rocci
    ```

    rocci is not on conda-forge at present, deliberately: the usual reason
    to want a conda build — coordinating shared native libraries across
    packages — does not apply here. The wheel is lightweight and
    self-contained (the Rust kernel is statically linked; the only
    dependencies are numpy and scipy), so pip-inside-conda is safe and is
    the supported path: pip reuses the conda-managed numpy/scipy when they
    satisfy the requirements and touches nothing else in the environment.
    If a conda-forge package would matter for your setup, open an issue —
    demand is what would change the decision.

=== "from source"

    ```bash
    # needs a Rust toolchain: https://rustup.rs
    pip install git+https://github.com/ndelaneybusch/rocci
    ```

    Installing from a git ref (or from the sdist on a platform with no
    wheel) compiles the Rust kernel locally via maturin. Build time is a
    couple of minutes; the result is identical to a released wheel.

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
