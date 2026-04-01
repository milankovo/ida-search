"""
Backends -- convert IR ``SearchTerm`` nodes into searchable queries.

Each backend targets a different search strategy, ordered from
fastest/least-semantic to slowest/most-semantic:

1. ``ByteSearchBackend``       -- raw byte matching via ``idaapi.bin_search``.
2. ``InsnOperandBackend``      -- ``decode_insn`` at every offset, inspect operands.
3. ``MicrocodeBackend``        -- Hex-Rays ``mba.for_all_ops`` microcode visitor.
4. ``CTreeBackend``            -- Hex-Rays ``cfunc_t.body`` ctree walk.
5. ``PseudocodeTextBackend``   -- substring search in decompiled pseudocode text.

Backends 1-2 are always available.  Backends 3-5 require the Hex-Rays
decompiler.

To add a new backend, define an ``emit`` method that accepts a
``SearchTerm`` and returns a query dataclass (or ``None`` if the term
type is not applicable).

IR term applicability::

    Backend               | Number | Text | Bytes | Float
    ----------------------|--------|------|-------|------
    ByteSearchBackend     |  yes   | yes  |  yes  |  yes
    InsnOperandBackend    |  yes   |  --  |   --  |   --
    MicrocodeBackend      |  yes   | yes  |   --  |  yes
    CTreeBackend          |  yes   | yes  |   --  |  yes
    PseudocodeTextBackend |  yes   | yes  |   --  |  yes
"""

from __future__ import annotations

import struct
import typing
from dataclasses import dataclass, field
from enum import Enum

from contextlib import contextmanager

from ir import SearchTerm, NumberTerm, TextTerm, BytesTerm, FloatTerm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@contextmanager
def wait_box(message: str):
    """Context manager wrapping ``idaapi.show_wait_box`` / ``hide_wait_box``."""
    import idaapi

    idaapi.show_wait_box(message)
    try:
        yield
    finally:
        idaapi.hide_wait_box()


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class Endian(Enum):
    LITTLE = "little"
    BIG = "big"
    BOTH = "both"

    @classmethod
    def from_ida(cls) -> Endian:
        import idaapi

        return cls("big" if idaapi.inf_is_be() else "little")


@dataclass(frozen=True, eq=True)
class Pattern:
    """A byte pattern with optional mask, ready for ``idaapi.bin_search``.

    When ``mask`` is not ``None``, bytes with mask ``0xFF`` must match
    exactly while bytes with mask ``0x00`` are wildcards.

    If ``strict_mask`` is ``True``, at least one mask byte has a value
    other than ``0x00`` or ``0xFF`` (per-bit masking), which requires
    ``BIN_SEARCH_BITMASK`` in IDA.

    ``source`` is an informational tag (e.g. the ``TypeSpec`` that
    produced the term) used for display purposes.
    """

    data: bytes
    mask: bytes | None = None
    strict_mask: bool = False
    source: typing.Any = None

    def __hash__(self):
        return hash((self.data, self.mask, self.strict_mask))

    def __eq__(self, other):
        return (
            self.data == other.data
            and self.mask == other.mask
            and self.strict_mask == other.strict_mask
        )

    def __str__(self):
        label = self.source.name if self.source else "?"
        if self.mask:
            assert len(self.data) == len(self.mask)
            parts: list[str] = []
            if self.strict_mask:
                for d, m in zip(self.data, self.mask):
                    parts.append(f"{d:02X}/{m:02X}")
            else:
                for d, m in zip(self.data, self.mask):
                    parts.append(f"{d:02X}" if m == 0xFF else "??")
            return f"{label}: {' '.join(parts)}"
        return f"{label}: {self.data.hex(' ')}"


# ===================================================================
# 1. ByteSearchBackend
# ===================================================================


def _encode_string(
    value: str, encodings: list[str]
) -> typing.Generator[bytes, None, None]:
    for enc in encodings:
        try:
            yield value.encode(enc)
        except (UnicodeEncodeError, LookupError):
            pass


class ByteSearchBackend:
    """Produce ``Pattern`` objects for ``idaapi.bin_search``."""

    def emit(
        self,
        term: SearchTerm,
        endian: Endian,
        encodings: list[str],
        *,
        source: typing.Any = None,
        case_sensitive: bool = False,
    ) -> list[Pattern]:
        match term:
            case NumberTerm():
                return self._emit_number(term, endian, source)
            case TextTerm():
                return self._emit_text(term, endian, encodings, source, case_sensitive)
            case BytesTerm():
                return self._emit_bytes(term, source)
            case FloatTerm():
                return self._emit_float(term, endian, source)
            case _:
                return []

    # -- Number -----------------------------------------------------------

    @staticmethod
    def _emit_number(
        term: NumberTerm, endian: Endian, source: typing.Any
    ) -> list[Pattern]:
        results: list[bytes] = []
        try:
            match endian:
                case Endian.LITTLE:
                    results.append(
                        term.value.to_bytes(term.width, "little", signed=term.signed)
                    )
                case Endian.BIG:
                    results.append(
                        term.value.to_bytes(term.width, "big", signed=term.signed)
                    )
                case Endian.BOTH:
                    results.append(
                        term.value.to_bytes(term.width, "little", signed=term.signed)
                    )
                    results.append(
                        term.value.to_bytes(term.width, "big", signed=term.signed)
                    )
        except OverflowError:
            return []
        return list({Pattern(data=r, source=source) for r in results})

    # -- Text -------------------------------------------------------------

    @staticmethod
    def _emit_text(
        term: TextTerm,
        endian: Endian,
        encodings: list[str],
        source: typing.Any,
        case_sensitive: bool = False,
    ) -> list[Pattern]:
        enc_list: list[str]
        match term.encoding:
            case None:
                enc_list = list(encodings)
            case "ascii+latin1":
                enc_list = ["ascii", "latin1"]
            case "unicode":
                enc_list = ["utf-8"]
                match endian:
                    case Endian.BIG:
                        enc_list.extend(["utf-16be", "utf-32be"])
                    case Endian.LITTLE:
                        enc_list.extend(["utf-16le", "utf-32le"])
                    case Endian.BOTH:
                        enc_list.extend(
                            ["utf-16le", "utf-16be", "utf-32le", "utf-32be"]
                        )
            case "cp500":
                enc_list = ["cp500"]
            case _:
                enc_list = [term.encoding]

        if case_sensitive:
            return list(
                {
                    Pattern(data=b, source=source)
                    for b in _encode_string(term.text, enc_list)
                }
            )

        # Case-insensitive: bitmask trick for ASCII-compatible encodings,
        # upper/lower variant fallback for EBCDIC.
        results: set[Pattern] = set()
        for enc in enc_list:
            if enc == "cp500":
                for variant in (term.text.upper(), term.text.lower()):
                    try:
                        results.add(Pattern(data=variant.encode(enc), source=source))
                    except (UnicodeEncodeError, LookupError):
                        pass
            else:
                try:
                    data = term.text.encode(enc)
                except (UnicodeEncodeError, LookupError):
                    continue
                mask_bytes = bytearray(len(data))
                has_alpha = False
                for i, b in enumerate(data):
                    if 0x41 <= (b & 0xDF) <= 0x5A:
                        mask_bytes[i] = 0xDF
                        has_alpha = True
                    else:
                        mask_bytes[i] = 0xFF
                if has_alpha:
                    results.add(
                        Pattern(
                            data=data,
                            mask=bytes(mask_bytes),
                            strict_mask=True,
                            source=source,
                        )
                    )
                else:
                    results.add(Pattern(data=data, source=source))
        return list(results)

    # -- Bytes ------------------------------------------------------------

    @staticmethod
    def _emit_bytes(term: BytesTerm, source: typing.Any) -> list[Pattern]:
        return [
            Pattern(
                data=term.data,
                mask=term.mask,
                strict_mask=term.strict_mask,
                source=source,
            )
        ]

    # -- Float ------------------------------------------------------------

    @staticmethod
    def _emit_float(
        term: FloatTerm, endian: Endian, source: typing.Any
    ) -> list[Pattern]:
        fmt = "f" if term.width == 4 else "d"
        endian_chars: list[str] = []
        match endian:
            case Endian.LITTLE:
                endian_chars = ["<"]
            case Endian.BIG:
                endian_chars = [">"]
            case Endian.BOTH:
                endian_chars = ["<", ">"]
        return list(
            {
                Pattern(data=struct.pack(f"{ec}{fmt}", term.value), source=source)
                for ec in endian_chars
            }
        )


# ===================================================================
# 2. InsnOperandBackend
# ===================================================================


@dataclass(frozen=True)
class OperandQuery:
    """Values to look for in decoded instruction operands."""

    values: frozenset[int]


class InsnOperandBackend:
    """Produce an ``OperandQuery`` for brute-force instruction decode sweep."""

    def emit(self, term: SearchTerm) -> OperandQuery | None:
        match term:
            case NumberTerm():
                return OperandQuery(values=frozenset({term.value}))
            case _:
                return None


def _segment_ranges(start_ea: int, end_ea: int):
    """Yield ``(seg_start, seg_end)`` for every segment intersecting the range."""
    import ida_segment

    seg = ida_segment.get_first_seg()
    while seg is not None:
        s = max(seg.start_ea, start_ea)
        e = min(seg.end_ea, end_ea)
        if s < e:
            yield s, e
        seg = ida_segment.get_next_seg(seg.start_ea)


def search_insn_operands(
    query: OperandQuery,
    start_ea: int,
    end_ea: int,
) -> typing.Generator[tuple[int, int], None, None]:
    """Decode instructions at every byte offset and yield matches.

    Yields ``(ea, insn_length)`` for each instruction whose operands
    contain one of the target values.

    Requires ``idaapi`` (runs inside IDA).
    """
    import idaapi

    with wait_box("Searching instruction operands..."):
        insn = idaapi.insn_t()
        for seg_start, seg_end in _segment_ranges(start_ea, end_ea):
            ea = seg_start
            while ea < seg_end:
                length = idaapi.decode_insn(insn, ea)
                if length > 0:
                    for op in insn.ops:
                        if op.type == idaapi.o_void:
                            break
                        if op.value in query.values or op.addr in query.values:
                            yield ea, length
                            break
                ea += 1
                if ea % 0x10000 == 0:
                    idaapi.show_addr(ea)
                    if idaapi.user_cancelled():
                        return


# ===================================================================
# 3. MicrocodeBackend
# ===================================================================


@dataclass(frozen=True)
class MicrocodeQuery:
    """What to search for in Hex-Rays microcode operands.

    At least one of ``values`` or ``text`` must be set.
    """

    values: list[int] = field(default_factory=list)
    text: str | None = None
    float_value: float | None = None
    case_sensitive: bool = False
    reqmat: int | None = None

    def to_rangeset(self):
        """Build an ``idaapi.rangeset_t`` for numeric value matching."""
        import idaapi

        rs = idaapi.rangeset_t()
        for v in self.values:
            rs.add(v, v + 1)
        return rs


class MicrocodeBackend:
    """Produce a ``MicrocodeQuery`` for Hex-Rays microcode constant/string search."""

    def emit(
        self, term: SearchTerm, *, case_sensitive: bool = False
    ) -> MicrocodeQuery | None:
        match term:
            case NumberTerm():
                return MicrocodeQuery(values=[term.value])
            case TextTerm():
                return MicrocodeQuery(text=term.text, case_sensitive=case_sensitive)
            case FloatTerm():
                return MicrocodeQuery(float_value=term.value)
            case _:
                return None


class _MicrocodeVisitor:
    """Microcode operand visitor for numeric and text searches.

    Requires ``idaapi`` at runtime.
    """

    @staticmethod
    def search_in_func(fnc_ea: int, query: MicrocodeQuery) -> list[tuple[int, str]]:
        """Search one function's microcode for *query*.

        Returns list of ``(ea, description)`` for each match.
        """
        import idaapi

        if not idaapi.init_hexrays_plugin():
            return []
        pfn = idaapi.get_func(fnc_ea)
        if not pfn:
            return []

        hf = idaapi.hexrays_failure_t()
        mbr = idaapi.mba_ranges_t(pfn)
        decomp_flags = idaapi.DECOMP_ALL_BLKS | idaapi.DECOMP_NO_WAIT
        reqmat = query.reqmat if query.reqmat is not None else idaapi.MMAT_LOCOPT
        mba = idaapi.gen_microcode(mbr, hf, None, decomp_flags, reqmat)
        if not mba:
            return []

        rangeset = query.to_rangeset() if query.values else None

        class Visitor(idaapi.mop_visitor_t):
            def __init__(self):
                super().__init__()
                self.results: list[tuple[int, str]] = []

            def visit_mop(
                self, op: idaapi.mop_t, tif: idaapi.tinfo_t, is_target: bool
            ) -> int:
                ea = self.curins.ea if self.curins else 0

                if rangeset:
                    numeric_values: set[int] = set()

                    if op.is_constant():
                        numeric_values.add(op.unsigned_value())
                    if op.a is not None:
                        if op.a.is_constant():
                            numeric_values.add(op.a.unsigned_value())
                    if op.g is not None:
                        numeric_values.add(op.g)
                    if op.nnn is not None:
                        numeric_values.add(op.nnn.value)
                    if op.f is not None and op.f.callee != idaapi.BADADDR:
                        numeric_values.add(op.f.callee)

                    for numeric_value in numeric_values:
                        if rangeset.contains(numeric_value):
                            self.results.append((ea, op.dstr()))
                            break

                if query.text:
                    text_candidates: list[str] = []
                    if op.helper:
                        text_candidates.append(op.helper)
                    if op.cstr:
                        text_candidates.append(op.cstr)
                    if op.f is not None and op.f.callee != idaapi.BADADDR:
                        callee_name = idaapi.get_name(op.f.callee)
                        if callee_name:
                            text_candidates.append(callee_name)

                    for candidate in text_candidates:
                        if query.case_sensitive:
                            match = query.text in candidate
                        else:
                            match = query.text.lower() in candidate.lower()
                        if match:
                            self.results.append((ea, candidate))
                            break

                if query.float_value is not None and op.fpc is not None:
                    fpvalue = op.fpc.fnum
                    if (
                        hasattr(fpvalue, "_float")
                        and fpvalue._float == query.float_value
                    ):
                        self.results.append((ea, op.dstr()))
                return 0

        visitor = Visitor()
        mba.for_all_ops(visitor)
        return visitor.results


def search_microcode(
    query: MicrocodeQuery,
) -> typing.Generator[tuple[int, str], None, None]:
    """Sweep all functions in the IDB for microcode matches.

    Yields ``(ea, description)`` for each hit.
    Shows a wait box and supports cancellation.
    """
    import idaapi
    import idautils

    if not idaapi.init_hexrays_plugin():
        return

    with wait_box("Searching microcode..."):
        for fnc_ea in idautils.Functions():
            idaapi.show_addr(fnc_ea)
            try:
                results = _MicrocodeVisitor.search_in_func(fnc_ea, query)
                yield from results
            except Exception:
                pass
            if idaapi.user_cancelled():
                return


# ===================================================================
# 4. CTreeBackend
# ===================================================================


@dataclass(frozen=True)
class CTreeQuery:
    """What to search for in the decompiled ctree."""

    number: int | None = None
    text: str | None = None
    float_value: float | None = None
    case_sensitive: bool = False
    cmat: int | None = None


class CTreeBackend:
    """Produce a ``CTreeQuery`` for Hex-Rays ctree AST walk."""

    def emit(
        self, term: SearchTerm, *, case_sensitive: bool = False
    ) -> CTreeQuery | None:
        match term:
            case NumberTerm():
                return CTreeQuery(number=term.value)
            case TextTerm():
                return CTreeQuery(text=term.text, case_sensitive=case_sensitive)
            case FloatTerm():
                return CTreeQuery(float_value=term.value)
            case _:
                return None


def search_ctree_in_func(fnc_ea: int, query: CTreeQuery) -> list[tuple[int, str]]:
    """Search one function's ctree for *query*.

    Returns list of ``(ea, description)`` for each match.
    """
    import idaapi

    if not idaapi.init_hexrays_plugin():
        return []

    class Visitor(idaapi.ctree_parentee_t):
        def __init__(self):
            super().__init__()
            self.results: list[tuple[int, str]] = []

        def _match_ea(self, expr: idaapi.cexpr_t) -> int:
            if expr.ea != idaapi.BADADDR:
                return expr.ea

            parent_expr = self.parent_expr()
            if parent_expr is not None and parent_expr.ea != idaapi.BADADDR:
                return parent_expr.ea

            parent_insn = self.parent_insn()
            if parent_insn is not None and parent_insn.ea != idaapi.BADADDR:
                return parent_insn.ea

            return fnc_ea

        def visit_expr(self, expr: idaapi.cexpr_t):
            ea = self._match_ea(expr)

            if query.number is not None and expr.op == idaapi.cot_num:
                if expr.n._value == query.number:
                    self.results.append((ea, f"num {query.number:#x}"))
            elif query.number is not None and expr.op == idaapi.cot_obj:
                if expr.obj_ea == query.number:
                    self.results.append((ea, f"obj_ea {query.number:#x}"))

            if query.text is not None:
                if expr.op == idaapi.cot_str:
                    if query.case_sensitive:
                        hit = query.text in expr.string
                    else:
                        hit = query.text.lower() in expr.string.lower()
                    if hit:
                        self.results.append((ea, f'str "{expr.string}"'))
                elif expr.op == idaapi.cot_obj:
                    name = idaapi.get_name(expr.obj_ea)
                    if name:
                        if query.case_sensitive:
                            hit = query.text in name
                        else:
                            hit = query.text.lower() in name.lower()
                        if hit:
                            self.results.append((ea, f"obj {name}"))

            if query.float_value is not None and expr.op == idaapi.cot_fnum:
                fv = expr.fpc.fnum
                if hasattr(fv, "_float"):
                    if fv._float == query.float_value:
                        self.results.append((ea, f"fnum {query.float_value}"))

            return 0

    class MaturityHooks(idaapi.Hexrays_Hooks):
        def __init__(self, target_cmat: int):
            super().__init__()
            self.target_cmat = target_cmat
            self.results: list[tuple[int, str]] = []
            self.seen_target = False

        def maturity(self, cfunc, new_maturity):
            if cfunc.entry_ea != fnc_ea or new_maturity != self.target_cmat:
                return 0

            visitor = Visitor()
            visitor.apply_to(cfunc.body, None)
            self.results.extend(visitor.results)
            self.seen_target = True
            return 0

    target_cmat = query.cmat if query.cmat is not None else idaapi.CMAT_FINAL
    hooks = MaturityHooks(target_cmat)
    hooks.hook()
    try:
        cfunc = idaapi.decompile(fnc_ea)
        if not cfunc:
            return []

        if not hooks.seen_target and target_cmat == idaapi.CMAT_FINAL:
            visitor = Visitor()
            visitor.apply_to(cfunc.body, None)
            hooks.results.extend(visitor.results)

        return hooks.results
    except Exception:
        return []
    finally:
        hooks.unhook()


def search_ctree(
    query: CTreeQuery,
) -> typing.Generator[tuple[int, str], None, None]:
    """Sweep all functions for ctree matches.

    Yields ``(ea, description)``.
    """
    import idaapi
    import idautils

    if not idaapi.init_hexrays_plugin():
        return

    with wait_box("Searching ctree..."):
        for fnc_ea in idautils.Functions():
            idaapi.show_addr(fnc_ea)
            try:
                yield from search_ctree_in_func(fnc_ea, query)
            except Exception:
                pass
            if idaapi.user_cancelled():
                return


# ===================================================================
# 5. PseudocodeTextBackend
# ===================================================================


@dataclass(frozen=True)
class PseudocodeQuery:
    """Substring to search for in pseudocode text."""

    substring: str
    case_sensitive: bool = False


class PseudocodeTextBackend:
    """Produce a ``PseudocodeQuery`` for decompiled text search."""

    def emit(
        self, term: SearchTerm, *, case_sensitive: bool = False
    ) -> PseudocodeQuery | None:
        match term:
            case NumberTerm():
                return PseudocodeQuery(substring=f"0x{term.value:X}")
            case TextTerm():
                return PseudocodeQuery(
                    substring=term.text, case_sensitive=case_sensitive
                )
            case FloatTerm():
                return PseudocodeQuery(substring=str(term.value))
            case _:
                return None


def search_pseudocode(
    query: PseudocodeQuery,
) -> typing.Generator[tuple[int, str, int], None, None]:
    """Sweep all functions for pseudocode text matches.

    Yields ``(func_ea, matching_line_text, line_number)``.
    """
    import idaapi
    import idautils

    if not idaapi.init_hexrays_plugin():
        return

    with wait_box("Searching pseudocode..."):
        for fnc_ea in idautils.Functions():
            idaapi.show_addr(fnc_ea)
            try:
                cfunc = idaapi.decompile(fnc_ea)
            except Exception:
                continue
            if not cfunc:
                continue

            for idx, sline in enumerate(cfunc.pseudocode):
                line_text = idaapi.tag_remove(sline.line)
                if query.case_sensitive:
                    hit = query.substring in line_text
                else:
                    hit = query.substring.lower() in line_text.lower()
                if hit:
                    yield fnc_ea, line_text, idx

            if idaapi.user_cancelled():
                return
