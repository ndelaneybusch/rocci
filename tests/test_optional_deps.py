"""Optional-dependency contract: numpy is the only runtime dependency.

The failure this guards is uniquely invisible: a single top-level ``import
scipy``, ``import matplotlib``, or ``import pandas`` added anywhere in rocci
passes the entire suite in the dev environment (all three are installed) while
breaking every user who installed plain ``rocci``. The in-process block in
``test_plotting.py`` can't catch it either, because it runs after rocci is
already imported. The only honest test is a clean subprocess where the packages
are made unimportable *before* rocci loads — which is what a meta-path blocker
does here.

Guaranteed. With scipy, matplotlib, and pandas blocked, the full non-plotting
pipeline works — envelope band, Working-Hotelling band (including its normality
diagnostics, which run entirely on ``rocci.special``), OvR, ``summary()``,
``at()``, ``band_area``, and a ``show_versions()`` that degrades to "not
installed" rather than crashing. Every entry point that genuinely needs an
optional package (``plot``, ``plot_diagnostics``, ``diagnostics=True``,
``to_dataframe``) fails as a ``RocciError`` naming the exact fix
(``rocci[plot]`` or ``pandas``), never as a raw ``ImportError`` traceback. And
the mechanism those guarantees rely on — laziness — is verified directly: with
the packages available, importing rocci and building a band still leaves them
out of ``sys.modules``.

Limitations. Only scipy, matplotlib, and pandas are blocked (the packages a dev
environment always carries); this is a contract on import structure and error
messaging, not on the plots those packages produce (that is
``test_plotting.py``).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

#: Installed before anything else: makes scipy/matplotlib/pandas unimportable,
#: the moral equivalent of an environment where they were never installed.
_BLOCKER = """
import sys

class _BlockOptionalDeps:
    _blocked = ("scipy", "matplotlib", "pandas")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self._blocked:
            raise ImportError(f"{name} is blocked: optional-dep contract test")
        return None

sys.meta_path.insert(0, _BlockOptionalDeps())
"""


def run_blocked(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a clean interpreter with scipy/matplotlib/pandas blocked."""
    code = _BLOCKER + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_full_pipeline_works_with_numpy_alone():
    # everything a non-plotting user touches must work: envelope path, WH
    # path, OvR, summary, at(), band_area, show_versions
    proc = run_blocked("""
        import warnings

        import numpy as np

        warnings.simplefilter("ignore")  # WH normality warning is irrelevant here
        import rocci

        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(40, int), np.ones(40, int)]
        s = np.r_[rng.normal(0, 1, 40), rng.normal(1.5, 1, 40)]

        band = rocci.roc_band(y, s, n_boot=300, random_state=0)
        assert band.method == "envelope"
        assert "AUC" in band.summary()
        lo, tp, up = band.at([0.1, 0.5, 0.9])
        assert (lo <= up).all()
        assert 0.0 < band.band_area < 1.0

        wh = rocci.roc_band(y, s, normal=True)
        assert wh.method == "working_hotelling"

        y3 = np.repeat([0, 1, 2], 30)
        s3 = rng.normal(0, 1, (90, 3))
        for j in range(3):
            s3[y3 == j, j] += 3.0
        bands = rocci.roc_band_ovr(y3, s3, n_boot=300, random_state=0)
        assert len(bands) == 3

        rocci.show_versions()
        print("PIPELINE-OK")
    """)
    assert proc.returncode == 0, proc.stderr
    assert "PIPELINE-OK" in proc.stdout
    # show_versions must degrade gracefully, not crash
    assert "scipy: not installed" in proc.stdout
    assert "matplotlib: not installed" in proc.stdout


def test_error_paths_are_actionable_without_optional_deps():
    # every entry point that needs an optional package must fail as a
    # RocciError naming the exact install fix, never an ImportError traceback
    proc = run_blocked("""
        import numpy as np

        import rocci
        from rocci._exceptions import RocciError

        rng = np.random.default_rng(1)
        y = np.r_[np.zeros(30, int), np.ones(30, int)]
        s = np.r_[rng.normal(0, 1, 30), rng.normal(1.5, 1, 30)]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            band = rocci.roc_band(y, s, n_boot=300, random_state=0)

        for call in (band.plot, band.plot_diagnostics):
            try:
                call()
            except RocciError as err:
                assert "rocci[plot]" in str(err), str(err)
            else:
                raise AssertionError(f"{call.__name__} did not raise RocciError")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rocci.roc_band(y, s, n_boot=300, diagnostics=True, random_state=0)
        except RocciError as err:
            assert "rocci[plot]" in str(err), str(err)
        else:
            raise AssertionError("diagnostics=True did not raise RocciError")

        try:
            band.to_dataframe()
        except RocciError as err:
            assert "pandas" in str(err), str(err)
        else:
            raise AssertionError("to_dataframe did not raise RocciError")

        print("ERRORS-OK")
    """)
    assert proc.returncode == 0, proc.stderr
    assert "ERRORS-OK" in proc.stdout


def test_import_and_band_construction_stay_lazy():
    # with the optional packages *available*, importing rocci and building a
    # band must still not import them — laziness is the mechanism the two
    # tests above rely on, and an eager import is invisible in a dev
    # environment without this check
    code = textwrap.dedent("""
        import sys

        import numpy as np

        import rocci

        rng = np.random.default_rng(2)
        y = np.r_[np.zeros(30, int), np.ones(30, int)]
        s = np.r_[rng.normal(0, 1, 30), rng.normal(1.5, 1, 30)]
        band = rocci.roc_band(y, s, n_boot=300, random_state=0)
        band.summary()
        band.at([0.5])

        leaked = [
            m for m in ("scipy", "matplotlib", "pandas") if m in sys.modules
        ]
        assert not leaked, f"eagerly imported: {leaked}"
        print("LAZY-OK")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "LAZY-OK" in proc.stdout
