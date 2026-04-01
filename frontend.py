"""
Frontend -- parsers that convert user text into IR nodes.

Each ``TypeSpec`` subclass knows how to parse a particular input format
(hex bytes, integers, text, etc.) and produce one or more ``SearchTerm``
IR nodes.  The parsing is *purely syntactic* -- no encoding, endianness,
or search-strategy decisions happen here.

To add a new input type:
    1. Subclass ``TypeSpec``.
    2. Set ``name``, ``aliases``, ``description``, ``category``, ``rank``.
    3. Implement ``parse(value: str) -> list[SearchTerm]``.
    4. The new class is auto-registered via ``TypeSpec.__init_subclass__``.

The module also provides the routing layer:
    - ``split_text_and_type(t)`` -- extract value and type specifier from
      user input like ``"0x100,i32"``.
    - ``Types`` -- sorted list of all registered ``TypeSpec`` subclasses.
    - ``AliasToType`` -- alias string -> ``TypeSpec`` lookup dict.
"""

from __future__ import annotations

from enum import IntFlag, auto

from ir import SearchTerm, NumberTerm, TextTerm, BytesTerm, FloatTerm


# ---------------------------------------------------------------------------
# Pattern categories (metadata on type specs, used by the UI)
# ---------------------------------------------------------------------------


class PatternCategory(IntFlag):
    text = auto()
    hex = auto()
    signed = auto()
    unsigned = auto()
    float = auto()
    magic = auto()


# ---------------------------------------------------------------------------
# TypeSpec base
# ---------------------------------------------------------------------------

_registry: list[type[TypeSpec]] = []


class TypeSpec:
    """Base class for all input-type parsers.

    Subclasses are auto-collected into ``_registry`` on definition.
    """

    name: str
    aliases: list[str | None]
    description: str
    category: PatternCategory = PatternCategory(0)
    rank: int = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _registry.append(cls)

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        """Parse *value* into one or more IR nodes.

        Must be overridden by every concrete subclass.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Text parsers
# ---------------------------------------------------------------------------


class TextSpec(TypeSpec):
    """Searches for a string using all IDB encodings."""

    aliases = ["t"]
    name = "Text"
    description = "searches for a string using all IDB encodings."
    category = PatternCategory.text
    rank = 1

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        return [TextTerm(text=value, encoding=None)]


class ASCIISpec(TypeSpec):
    """Searches for an ASCII+ANSI string."""

    aliases = ["a"]
    name = "ASCII String"
    description = "searches for an ASCII+ANSI string."
    category = PatternCategory.text
    rank = 2

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        return [TextTerm(text=value, encoding="ascii+latin1")]


class UnicodeSpec(TypeSpec):
    """Searches for a Unicode string."""

    aliases = ["u"]
    name = "Unicode String"
    description = "searches for a Unicode string."
    category = PatternCategory.text
    rank = 3

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        return [TextTerm(text=value, encoding="unicode")]


class EBCDICSpec(TypeSpec):
    """Searches for an EBCDIC string."""

    aliases = ["e"]
    name = "EBCDIC String"
    description = "searches for an EBCDIC string."
    category = PatternCategory.text
    rank = 5

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        return [TextTerm(text=value, encoding="cp500")]


# ---------------------------------------------------------------------------
# Hex / masked byte parsers
# ---------------------------------------------------------------------------


class HexBytesSpec(TypeSpec):
    r"""Searches for a set of hex bytes.

    Hex bytes can be entered without spaces.  Separators like commas,
    semicolons, curly braces, and ``0x`` prefixes are stripped.
    Use ``??`` for wildcard bytes.

    Examples::

        453f,h          -> BytesTerm(b'\x45\x3f')
        45 ?? 3f,h      -> BytesTerm(b'\x45\x00\x3f', mask=b'\xff\x00\xff')
        0x41 0x42,h     -> BytesTerm(b'\x41\x42')
    """

    aliases = ["h"]
    name = "Hex Bytes"
    description = (
        "Searches for a set of hex bytes.  Use ?? for wildcard bytes.  "
        "Bytes can be entered without spaces."
    )
    category = PatternCategory.hex
    rank = 1

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        value = value.lower()
        for ch in ";{}":
            value = value.replace(ch, " ")
        value = value.replace(",", " ")
        value = value.replace("0x", "")
        while "  " in value:
            value = value.replace("  ", " ")
        value = value.strip()

        if "??" in value:
            parts = value.split("??")
            in_bytes = [bytes.fromhex(x) for x in parts]
            masks = [bytes.fromhex("FF" * len(x)) for x in in_bytes]
            pattern = b"\x00".join(in_bytes)
            mask = b"\x00".join(masks)
            return [BytesTerm(data=pattern, mask=mask)]

        return [BytesTerm(data=bytes.fromhex(value))]


class MaskedBytesSpec(TypeSpec):
    r"""Searches for bytes with per-bit masks.

    Every byte must be separated by a space.  Use ``?`` in place of a
    digit to mask that position.  Supports ``0b`` (binary), ``0o``
    (octal), and plain hex.  Alternatively, use ``byte/mask`` pairs.

    Grammar::

        value = byte (SP byte)*
        byte  = [prefix] digits | digits "/" digits
        prefix = "0b" | "0o"

    Examples::

        0b1?00001       -> matches 'A' or 'a'
        0x41/0x20       -> same (explicit mask)
    """

    aliases = ["m"]
    name = "Masked Bytes"
    description = (
        "Searches for bytes with per-bit masks.  Use ? for masked digits.  "
        "Every byte must be separated by a space."
    )
    category = PatternCategory.hex
    rank = 2

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        value = value.replace(",", " ")
        while "  " in value:
            value = value.replace("  ", " ")
        value = value.strip()
        parts = value.split(" ")

        in_bytes: list[int] = []
        mask: list[int] = []

        def to_byte(part: str) -> int:
            if part.startswith("0o"):
                return int(part[2:], 8)
            if part.startswith("0b"):
                return int(part[2:], 2)
            return int(part, 16)

        def _maskit(part: str, lo: str, hi: str) -> str:
            return "".join(hi if c == "?" else lo for c in part)

        def to_mask2(part: str, lo: str, hi: str) -> int:
            if part.startswith("0o"):
                return int(_maskit(part[2:], lo, hi), 8)
            if part.startswith("0b"):
                return int(_maskit(part[2:], lo, hi), 2)
            return int(_maskit(part, lo, hi), 16)

        def to_mask(part: str, lo: str, hi: str) -> int:
            return ~to_mask2(part, lo, hi) & 0xFF

        for p in parts:
            if "/" in p:
                byte_val, mask_val = p.split("/", 1)
                in_bytes.append(to_byte(byte_val.replace("?", "0")))
                mask.append(~to_byte(mask_val) & 0xFF)
            elif p.startswith("0o"):
                in_bytes.append(to_byte(p.replace("?", "0")))
                mask.append(to_mask(p, "0", "7"))
            elif p.startswith("0b"):
                in_bytes.append(to_byte(p.replace("?", "0")))
                mask.append(to_mask(p, "0", "1"))
            else:
                in_bytes.append(to_byte(p.replace("?", "0")))
                mask.append(to_mask(p, "0", "F"))

        if all(m == 0xFF for m in mask):
            return [BytesTerm(data=bytes(in_bytes))]

        strict = any(m != 0 and m != 0xFF for m in mask)
        return [BytesTerm(data=bytes(in_bytes), mask=bytes(mask), strict_mask=strict)]


# ---------------------------------------------------------------------------
# Integer parsers  (generated via a factory to avoid 20 near-identical classes)
# ---------------------------------------------------------------------------


def _make_int_spec(
    name: str,
    aliases: list[str],
    description: str,
    width: int,
    signed: bool,
    category: PatternCategory,
    rank: int,
) -> type[TypeSpec]:
    """Dynamically create a TypeSpec subclass for an integer type."""

    cls_dict = {
        "name": name,
        "aliases": aliases,
        "description": description,
        "category": category,
        "rank": rank,
    }

    def parse(cls_, value: str) -> list[SearchTerm]:
        n = int(value, 0)
        return [NumberTerm(value=n, width=width, signed=signed)]

    cls_dict["parse"] = classmethod(parse)
    return type(name.replace(" ", ""), (TypeSpec,), cls_dict)


SignedByteSpec = _make_int_spec(
    "Signed Byte",
    ["i8"],
    "searches for a signed byte.",
    1,
    True,
    PatternCategory.signed,
    1,
)
UnsignedByteSpec = _make_int_spec(
    "Unsigned Byte",
    ["u8", "db"],
    "searches for an unsigned byte.",
    1,
    False,
    PatternCategory.unsigned,
    1,
)
SignedShortSpec = _make_int_spec(
    "Signed Short",
    ["i16"],
    "searches for a signed short.",
    2,
    True,
    PatternCategory.signed,
    2,
)
UnsignedShortSpec = _make_int_spec(
    "Unsigned Short",
    ["u16", "dw"],
    "searches for an unsigned short.",
    2,
    False,
    PatternCategory.unsigned,
    2,
)
SignedIntSpec = _make_int_spec(
    "Signed Int",
    ["i32"],
    "searches for a signed dword.",
    4,
    True,
    PatternCategory.signed,
    3,
)
UnsignedIntSpec = _make_int_spec(
    "Unsigned Int",
    ["u32", "dd"],
    "searches for an unsigned dword.",
    4,
    False,
    PatternCategory.unsigned,
    3,
)
SignedQuadSpec = _make_int_spec(
    "Signed Quad",
    ["i64"],
    "searches for a signed quad word.",
    8,
    True,
    PatternCategory.signed,
    4,
)
UnsignedQuadSpec = _make_int_spec(
    "Unsigned Quad",
    ["u64", "dq"],
    "searches for an unsigned quad word.",
    8,
    False,
    PatternCategory.unsigned,
    4,
)


class AddressSpec(TypeSpec):
    """Searches for a hexadecimal address value."""

    aliases = ["addr", "ea", "address"]
    name = "Address"
    description = "searches for a hexadecimal address value."
    category = PatternCategory.unsigned
    rank = 4

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        normalized = value.strip().lower().replace("_", "")
        sign = -1 if normalized.startswith("-") else 1
        magnitude = normalized.lstrip("+-")
        if magnitude.startswith("0x"):
            magnitude = magnitude[2:]
        if not magnitude:
            raise ValueError("Address value is empty")

        n = int(value, 16)
        width = max(1, (len(magnitude) + 1) // 2)
        return [NumberTerm(value=n, width=width, signed=(sign < 0))]


SignedOctaSpec = _make_int_spec(
    "Signed Octa",
    ["i128"],
    "searches for a signed octaword.",
    16,
    True,
    PatternCategory.signed,
    5,
)
UnsignedOctaSpec = _make_int_spec(
    "Unsigned Octa",
    ["u128", "xmm"],
    "searches for an unsigned octaword.",
    16,
    False,
    PatternCategory.unsigned,
    5,
)
SignedHexadecaSpec = _make_int_spec(
    "Signed ymm word",
    ["i256"],
    "searches for a signed ymm word.",
    32,
    True,
    PatternCategory.signed,
    6,
)
UnsignedHexadecaSpec = _make_int_spec(
    "Unsigned ymm word",
    ["u256", "ymm"],
    "searches for an unsigned ymm word.",
    32,
    False,
    PatternCategory.unsigned,
    6,
)
SignedTetraSpec = _make_int_spec(
    "Signed zmm word",
    ["i512"],
    "searches for a signed zmm word.",
    64,
    True,
    PatternCategory.signed,
    7,
)
UnsignedTetraSpec = _make_int_spec(
    "Unsigned zmm word",
    ["u512", "zmm"],
    "searches for an unsigned zmm word.",
    64,
    False,
    PatternCategory.unsigned,
    7,
)


# ---------------------------------------------------------------------------
# Float parsers
# ---------------------------------------------------------------------------


class FloatSpec(TypeSpec):
    """Searches for a 32-bit float."""

    aliases = ["f", "f32", "float"]
    name = "Float"
    description = "searches for a float."
    category = PatternCategory.float
    rank = 1

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        return [FloatTerm(value=float(value), width=4)]


class DoubleSpec(TypeSpec):
    """Searches for a 64-bit double."""

    aliases = ["lf", "f64", "double"]
    name = "Double"
    description = "searches for a double."
    category = PatternCategory.float
    rank = 2

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        return [FloatTerm(value=float(value), width=8)]


# ---------------------------------------------------------------------------
# Magic (auto-detect)
# ---------------------------------------------------------------------------


class MagicSpec(TypeSpec):
    """Tries every parser with rank < 5 and returns all successful results."""

    aliases = ["magic", "", None]
    name = "Magic"
    description = "Searches for any type with rank < 5."
    category = PatternCategory.magic
    rank = 10000

    @classmethod
    def parse(cls, value: str) -> list[SearchTerm]:
        terms: list[SearchTerm] = []
        for spec in Types:
            if spec is cls or spec.rank >= 5:
                continue
            try:
                terms.extend(spec.parse(value))
            except Exception:
                pass
        return terms


# ---------------------------------------------------------------------------
# Registry and routing
# ---------------------------------------------------------------------------

Types: list[type[TypeSpec]] = sorted(
    _registry, key=lambda x: (x.category, x.rank, x.name)
)

AliasToType: dict[str | None, type[TypeSpec]] = {
    alias: t for t in Types for alias in t.aliases
}


def split_text_and_type(t: str) -> tuple[str, type[TypeSpec]]:
    """Split ``"value,type_alias"`` into ``(value, TypeSpec class)``.

    If no comma is present the type defaults to ``MagicSpec``.

    Raises ``ValueError`` for unknown type aliases.
    """
    t = t.rstrip()
    if "," in t:
        value, type_alias = t.rsplit(",", 1)
    else:
        value, type_alias = t, None

    cls = AliasToType.get(type_alias)
    if cls is None:
        raise ValueError(f"Unknown type specifier: {type_alias!r}")
    return value, cls
