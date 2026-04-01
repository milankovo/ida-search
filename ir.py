"""
Intermediate Representation for IDA Search.

This module defines the semantic IR nodes that sit between the frontend (text
parsing) and the backends (byte search, instruction operand scan, microcode
visitor, ctree walk, pseudocode text search).

Each node captures *what the user meant* without committing to a particular
search strategy.  A ``NumberTerm`` says "the user typed the integer 256 as a
signed 32-bit value" -- it is up to a backend to decide whether to encode that
as little-endian bytes, look for it in instruction operands, etc.

To add a new semantic category, subclass ``SearchTerm`` as a frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchTerm:
    """Base class for all IR nodes."""


@dataclass(frozen=True)
class NumberTerm(SearchTerm):
    """User typed a numeric value.

    Attributes:
        value: The integer value.
        width: Size in bytes (1, 2, 4, 8, 16, 32, 64).
        signed: Whether the value is signed.
    """

    value: int
    width: int
    signed: bool


@dataclass(frozen=True)
class TextTerm(SearchTerm):
    """User typed a text string.

    Attributes:
        text: The string to search for.
        encoding: Encoding hint.  ``None`` means "use all available encodings".
            Specific values like ``"cp500"`` or ``"unicode"`` narrow the set.
    """

    text: str
    encoding: str | None = None


@dataclass(frozen=True)
class BytesTerm(SearchTerm):
    """User typed raw byte patterns (hex bytes or masked bytes).

    Attributes:
        data: The byte values.
        mask: Per-byte mask (``None`` means match all bytes exactly).
            ``0xFF`` in a mask position means "must match", ``0x00`` means
            "don't care".
        strict_mask: ``True`` when any mask byte is neither ``0x00`` nor
            ``0xFF`` (i.e. individual bits are masked).  This triggers
            ``BIN_SEARCH_BITMASK`` in IDA.
    """

    data: bytes
    mask: bytes | None = None
    strict_mask: bool = False


@dataclass(frozen=True)
class FloatTerm(SearchTerm):
    """User typed a floating-point number.

    Attributes:
        value: The float/double value.
        width: 4 for ``float``, 8 for ``double``.
    """

    value: float
    width: int
