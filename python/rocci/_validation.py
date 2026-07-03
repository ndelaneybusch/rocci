"""Shared argument checks for the public API.

Small, side-effecting validators that translate user-facing arguments into the
internal quantities the pipeline needs (``alpha``, an RNG seed), emitting the
warning taxonomy on out-of-range inputs. Kept separate from
:mod:`rocci._api` so the orchestration reads as a straight pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np

from rocci._exceptions import RocciError
from rocci._warnings import LowConfidenceWarning, RocciWarning


def check_confidence(confidence: float) -> float:
    """Validate ``confidence`` and return the significance level ``alpha``.

    Args:
        confidence: Simultaneous coverage target, in ``(0, 1)``.

    Returns:
        ``alpha = 1 - confidence``.

    Raises:
        RocciError: If ``confidence`` is not strictly inside ``(0, 1)``.

    Warns:
        LowConfidenceWarning: If ``confidence < 0.90`` — sup-norm bands
            intentionally over-cover at low levels.

    Examples:
        >>> from rocci._validation import check_confidence
        >>> round(check_confidence(0.95), 4)
        0.05
    """
    if not np.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise RocciError(
            f"confidence must be a number strictly inside (0, 1), got {confidence!r}. "
            "Pass e.g. confidence=0.95 for a 95% band."
        )
    if confidence < 0.90:
        warnings.warn(
            f"confidence={confidence:.2f} < 0.90: sup-norm simultaneous bands "
            "intentionally over-cover at low confidence levels, so the band will "
            "be wider than the nominal level suggests.",
            LowConfidenceWarning,
            stacklevel=3,
        )
    return 1.0 - confidence


def check_n_boot(n_boot: int) -> None:
    """Validate the bootstrap replicate count.

    Args:
        n_boot: Number of bootstrap replicates.

    Raises:
        RocciError: If ``n_boot < 100`` — too few for a usable quantile.

    Warns:
        RocciWarning: If ``n_boot < 1000`` — coarse quantile resolution.

    Examples:
        >>> from rocci._validation import check_n_boot
        >>> check_n_boot(2000) is None
        True
    """
    if n_boot < 100:
        raise RocciError(
            f"n_boot must be >= 100 for a usable bootstrap quantile, got {n_boot}. "
            "The default of 2000 is a good starting point."
        )
    if n_boot < 1000:
        warnings.warn(
            f"n_boot={n_boot} < 1000 gives coarse quantile resolution; consider "
            "n_boot >= 1000 (default 2000) for a stable band.",
            RocciWarning,
            stacklevel=3,
        )


def resolve_seed(random_state: int | None) -> int:
    """Turn ``random_state`` into a concrete non-negative kernel seed.

    ``None`` draws fresh entropy (a non-reproducible run); an integer is used
    as-is so that the same seed yields a bit-identical band on one backend.
    The value stored on :class:`~rocci._result.RocBand` is the *original*
    ``random_state`` (which may be ``None``), not this resolved seed.

    Args:
        random_state: User seed or ``None``.

    Returns:
        A non-negative integer below ``2**64``.

    Raises:
        RocciError: If ``random_state`` is a negative integer.

    Examples:
        >>> from rocci._validation import resolve_seed
        >>> resolve_seed(0)
        0
    """
    if random_state is None:
        return int(np.random.default_rng().integers(0, 2**64, dtype=np.uint64))
    seed = int(random_state)
    if seed < 0:
        raise RocciError(
            "random_state must be a non-negative integer or None, got "
            f"{random_state!r}."
        )
    return seed
