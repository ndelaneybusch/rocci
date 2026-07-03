"""Typed exceptions for rocci.

Every input error raised by rocci is a :class:`RocciError`, with a message
that states the fix, not just the failure.
"""


class RocciError(ValueError):
    """Base class for all rocci input and usage errors.

    Subclasses ``ValueError`` so that generic error handling around
    array-validation code keeps working, while letting callers catch
    rocci-specific failures precisely.

    Examples:
        >>> from rocci._exceptions import RocciError
        >>> try:
        ...     raise RocciError("pass pos_label= to disambiguate")
        ... except ValueError as err:
        ...     print(err)
        pass pos_label= to disambiguate
    """
