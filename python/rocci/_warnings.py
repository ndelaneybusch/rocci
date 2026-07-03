"""Warning taxonomy for rocci.

All rocci warnings derive from :class:`RocciWarning` so users can silence or
escalate the whole family with one ``filterwarnings`` rule.
"""


class RocciWarning(UserWarning):
    """Base class for all rocci warnings.

    Examples:
        >>> import warnings
        >>> from rocci._warnings import RocciWarning, TiesWarning
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     warnings.warn("heavy ties", TiesWarning, stacklevel=1)
        >>> issubclass(caught[0].category, RocciWarning)
        True
    """


class NormalityWarning(RocciWarning):
    """Binormality looks doubtful for the Working-Hotelling band."""


class LowConfidenceWarning(RocciWarning):
    """Confidence below 0.90: sup-norm bands intentionally over-cover there."""


class SmallSampleWarning(RocciWarning):
    """A class has fewer than 20 samples; exact floors dominate the band."""


class TiesWarning(RocciWarning):
    """Scores are heavily tied; the band stays valid but conservative."""


class FallbackBackendWarning(RocciWarning):
    """The Rust core is missing; the slower NumPy fallback kernel is active."""
