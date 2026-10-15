"""Errors raised by the Ikarus adapter and index lifecycle."""


class IkarusIndexError(RuntimeError):
    """An Ikarus-backed Bibliograph index is absent or incompatible."""
