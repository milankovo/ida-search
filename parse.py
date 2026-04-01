"""
Orchestration layer and public API for IDA Search.

This module ties together the three pipeline stages:

    Frontend (``frontend.py``)  →  IR (``ir.py``)  →  Backend (``backend.py``)

It provides ``PatternLocator`` -- the main entry point that both
``plugin.py`` and ``ask_form.py`` use -- as well as backward-compatible
re-exports of ``Types``, ``AliasToType``, ``Pattern``, ``PatternCategory``,
and the ``help()`` / ``overview()`` / ``html_overview()`` display helpers.
"""

from __future__ import annotations

from itertools import groupby

from ir import SearchTerm, NumberTerm, TextTerm, BytesTerm, FloatTerm
from frontend import (
    TypeSpec,
    Types,
    AliasToType,
    PatternCategory,
    split_text_and_type,
    UnicodeSpec,
)
from backend import (
    Endian,
    Pattern,
    ByteSearchBackend,
    InsnOperandBackend,
    OperandQuery,
    MicrocodeBackend,
    MicrocodeQuery,
    CTreeBackend,
    CTreeQuery,
    PseudocodeTextBackend,
    PseudocodeQuery,
    search_insn_operands,
    search_microcode,
    search_ctree,
    search_pseudocode,
)

# Re-export for backward compatibility (ask_form.py references parse.UnicodeString)
UnicodeString = UnicodeSpec

# ---------------------------------------------------------------------------
# PatternLocator -- main entry point
# ---------------------------------------------------------------------------


class PatternLocator:
    """Bridges user input to backend queries.

    Typical usage::

        loc = PatternLocator.from_string("0x100,i32", endian="little")
        patterns = loc.to_pattern()          # byte-search patterns
        oq = loc.to_operand_query()          # instruction operand query (or None)
        mq = loc.to_microcode_query()        # microcode query (or None)
        cq = loc.to_ctree_query()            # ctree query (or None)
        pq = loc.to_pseudocode_query()       # pseudocode query (or None)
    """

    __slots__ = ("value", "type_spec", "endian", "encodings", "case_sensitive")

    def __init__(
        self,
        value: str,
        type_spec: type[TypeSpec],
        endian: Endian = Endian.LITTLE,
        encodings: tuple[str, ...] = ("utf-8", "utf-16le", "utf-32le"),
        case_sensitive: bool = False,
    ):
        self.value = value
        self.type_spec = type_spec
        self.endian = endian
        self.encodings = encodings
        self.case_sensitive = case_sensitive

    # -- Backward-compatible aliases --
    @property
    def type(self):
        return self.type_spec

    def __str__(self):
        return f"{self.value},{self.type_spec.name}"

    def __repr__(self):
        return f"{self.value},{self.type_spec.name}"

    def __eq__(self, other):
        return (
            self.value == other.value
            and self.type_spec == other.type_spec
            and self.endian == other.endian
            and self.encodings == other.encodings
            and self.case_sensitive == other.case_sensitive
        )

    def __hash__(self):
        return hash((self.value, self.type_spec))

    @classmethod
    def from_string(cls, t: str, endian="little", encodings=None, case_sensitive=False):
        """Parse a user query string like ``"0x100,i32"``."""
        value, type_spec = split_text_and_type(t)
        if encodings is None:
            encodings = ("utf-8", "utf-16le", "utf-32le")
        if isinstance(encodings, list):
            encodings = tuple(encodings)
        if isinstance(endian, str):
            endian = Endian(endian)
        return cls(value, type_spec, endian, encodings, case_sensitive)

    def to_string(self):
        aliases = [a for a in self.type_spec.aliases if a]
        return f"{self.value},{aliases[0]}" if aliases else self.value

    # -- Frontend: parse to IR --

    def to_ir(self) -> list[SearchTerm]:
        """Parse user input into semantic IR nodes."""
        return self.type_spec.parse(self.value)

    # -- Backend 1: byte-level patterns --

    def to_pattern(self) -> list[Pattern]:
        """Produce byte-search ``Pattern`` objects (existing behavior)."""
        backend = ByteSearchBackend()
        terms = self.to_ir()
        patterns: list[Pattern] = []
        for term in terms:
            patterns.extend(
                backend.emit(
                    term, self.endian, list(self.encodings),
                    source=self.type_spec, case_sensitive=self.case_sensitive,
                )
            )
        return list(set(patterns))

    # -- Backend 2: instruction operand query --

    def to_operand_query(self) -> OperandQuery | None:
        """Produce an instruction-operand query, or ``None`` if not applicable."""
        backend = InsnOperandBackend()
        for term in self.to_ir():
            result = backend.emit(term)
            if result is not None:
                return result
        return None

    # -- Backend 3: microcode query --

    def to_microcode_query(self, reqmat: int | None = None) -> MicrocodeQuery | None:
        """Produce a Hex-Rays microcode query, or ``None`` if not applicable."""
        backend = MicrocodeBackend()
        for term in self.to_ir():
            result = backend.emit(term, case_sensitive=self.case_sensitive)
            if result is not None:
                if reqmat is not None:
                    return MicrocodeQuery(
                        values=result.values,
                        text=result.text,
                        float_value=result.float_value,
                        case_sensitive=result.case_sensitive,
                        reqmat=reqmat,
                    )
                return result
        return None

    # -- Backend 4: ctree query --

    def to_ctree_query(self, cmat: int | None = None) -> CTreeQuery | None:
        """Produce a Hex-Rays ctree query, or ``None`` if not applicable."""
        backend = CTreeBackend()
        for term in self.to_ir():
            result = backend.emit(term, case_sensitive=self.case_sensitive)
            if result is not None:
                if cmat is not None:
                    return CTreeQuery(
                        number=result.number,
                        text=result.text,
                        float_value=result.float_value,
                        case_sensitive=result.case_sensitive,
                        cmat=cmat,
                    )
                return result
        return None

    # -- Backend 5: pseudocode text query --

    def to_pseudocode_query(self) -> PseudocodeQuery | None:
        """Produce a pseudocode text query, or ``None`` if not applicable."""
        backend = PseudocodeTextBackend()
        for term in self.to_ir():
            result = backend.emit(term, case_sensitive=self.case_sensitive)
            if result is not None:
                return result
        return None


# ---------------------------------------------------------------------------
# Display helpers (used by ask_form.py and the help system)
# ---------------------------------------------------------------------------


def help() -> str:
    """Full help text listing all type specifiers."""
    text = (
        'Specify type by placing a comma followed by a type specifier '
        'at the end of the value to find in the text field.\n'
        'For example, the value "453f,h" would search for the hex bytes '
        '0x45 and 0x3f, the value "0x100,i32" would search for the '
        'integer 256, or the value "2.5,lf" would search for the '
        "double '2.5'.\n\n"
        'Rank is an internal priority used by Magic mode: lower-rank types '
        'are considered first, and Magic currently only tries types with '
        'rank < 5.\n\n'
        "The full list of type specifiers is:"
    )
    for t in sorted(Types, key=lambda x: (x.category, x.rank, x.name)):
        aliases = [a for a in t.aliases if a]
        text += f"\n{t.name} ({','.join(aliases)}) - {t.description}"
    return text


def overview() -> str:
    """Short text overview grouped by category."""
    lines: list[str] = []
    for feature, types_iter in groupby(
        sorted(Types, key=lambda x: (x.category, x.rank, x.name)),
        key=lambda x: x.category,
    ):
        aliases: list[str] = []
        for t in types_iter:
            aliases.extend(a for a in t.aliases if a)
        lines.append(f"{feature.name}: {', '.join(aliases)}")
    return "\n".join(lines)


def html_overview() -> str:
    """HTML table overview grouped by category."""
    rows: list[str] = []
    for feature, types_iter in groupby(
        sorted(Types, key=lambda x: (x.category, x.rank, x.name)),
        key=lambda x: x.category,
    ):
        aliases: list[str] = []
        for t in types_iter:
            aliases.extend(a for a in t.aliases if a)
        rows.append(f"<tr><td>{feature.name}</td><td>{', '.join(aliases)}</td></tr>")

    nl = "\n"
    return f"""
<table>
    <tr><th>Feature</th><th>Aliases</th></tr>
    {nl.join(rows)}
</table>"""


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

def _test_split_text_and_type():
    from frontend import HexBytesSpec, MagicSpec
    assert split_text_and_type("453f,h") == ("453f", HexBytesSpec)
    assert split_text_and_type("0x100,i32")[0] == "0x100"
    assert split_text_and_type("2.5,lf")[0] == "2.5"
    assert split_text_and_type("2.5")[1] is MagicSpec
    print("split_text_and_type tests PASSED")


def main():
    _test_split_text_and_type()

    while True:
        t = input("Enter a value to search: ")
        if t == "q":
            break

        p = PatternLocator.from_string(t)
        print(f"IR: {p.to_ir()}")
        v = p.to_pattern()
        print(f"{p!r}")
        for pat in v:
            print(str(pat))


if __name__ == "__main__":
    print(help())
    print(html_overview())
    main()
