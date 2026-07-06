# Installation

```bash
pip install rocci            # core: numpy + scipy only
pip install 'rocci[plot]'    # optional plotting support
```

Binary wheels ship for Linux (x86-64, aarch64, musl), macOS (Intel and Apple
silicon), and Windows, for every Python ≥ 3.10. This means that a `pip install
rocci` never needs to compile the rust components on a supported platform. 

## Supported install methods

=== "pip"

    ```bash
    pip install 'rocci[plot]'
    pip install rocci==0.2.0        # pin for reproducible environments
    ```

=== "uv"

    ```bash
    uv add rocci                    # core: numpy + scipy only
    uv pip install 'rocci[plot]'    # optional plotting support
    ```

=== "conda"

    ```bash
    conda activate myenv
    pip install rocci
    ```

    rocci is not on conda-forge at present. The usual reason
    to want a conda build is to coordinate shared native libraries across
    packages. That does not apply here. The wheel is lightweight and
    self-contained (the Rust kernel is statically linked; the only
    dependencies are numpy and scipy). Calling pip-inside-conda should suffice
     for most users. If a conda-forge package would matter for your setup, 
     open an issue — demand is what would change the decision.

=== "from source"

    ```bash
    # needs a Rust toolchain: https://rustup.rs
    pip install git+https://github.com/ndelaneybusch/rocci
    ```

    Installing from a git ref (or from the sdist on a platform with no
    wheel) compiles the Rust kernel locally via maturin. Build time is a
    couple of minutes.

## The fallback backend

rocci's bootstrap kernel is compiled Rust. On a platform with no wheel and no
Rust toolchain, the package still works: a pure-NumPy kernel with identical
statistical semantics takes over automatically, and rocci emits a
`FallbackBackendWarning` once per process so you know you are on the slow
path (10–30× slower). Which backend is active is recorded on every result as
`band.backend` and printed by `rocci.show_versions()`.

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
