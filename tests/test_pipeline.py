"""Comprehensive tests for the IDA Search pipeline.

Covers IR nodes, frontend parsers, backend emit logic, and orchestration.
All tests run outside IDA -- no ``idaapi`` dependency.
"""

from __future__ import annotations

import struct

import pytest

from ir import SearchTerm, NumberTerm, TextTerm, BytesTerm, FloatTerm, RangeTerm
from frontend import (
    TypeSpec,
    Types,
    AliasToType,
    PatternCategory,
    split_text_and_type,
    HexBytesSpec,
    MaskedBytesSpec,
    TextSpec,
    ASCIISpec,
    UnicodeSpec,
    EBCDICSpec,
    FloatSpec,
    DoubleSpec,
    MagicSpec,
    SignedIntSpec,
    UnsignedIntSpec,
    AddressSpec,
    SignedByteSpec,
    UnsignedByteSpec,
    SignedShortSpec,
    UnsignedShortSpec,
    SignedQuadSpec,
    UnsignedQuadSpec,
    RangeSpec,
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
    _iter_ctree_numeric_values,
    _iter_switch_case_values,
)
from parse import PatternLocator, help, overview, html_overview


# ===================================================================
# IR node tests
# ===================================================================


class TestIRNodes:
    def test_number_term_frozen(self):
        t = NumberTerm(value=256, width=4, signed=True)
        assert t.value == 256
        assert t.width == 4
        assert t.signed is True
        with pytest.raises(AttributeError):
            t.value = 0

    def test_text_term_frozen(self):
        t = TextTerm(text="hello", encoding="ascii+latin1")
        assert t.text == "hello"
        assert t.encoding == "ascii+latin1"
        with pytest.raises(AttributeError):
            t.text = "x"

    def test_text_term_default_encoding(self):
        t = TextTerm(text="abc")
        assert t.encoding is None

    def test_bytes_term_frozen(self):
        t = BytesTerm(data=b"\xde\xad", mask=b"\xff\xff")
        assert t.data == b"\xde\xad"
        assert t.mask == b"\xff\xff"
        assert t.strict_mask is False

    def test_bytes_term_no_mask(self):
        t = BytesTerm(data=b"\x90")
        assert t.mask is None

    def test_float_term_frozen(self):
        t = FloatTerm(value=2.5, width=4)
        assert t.value == 2.5
        assert t.width == 4
        with pytest.raises(AttributeError):
            t.width = 8

    def test_range_term_frozen(self):
        t = RangeTerm(low=100, high=200)
        assert t.low == 100
        assert t.high == 200
        with pytest.raises(AttributeError):
            t.low = 0

    def test_all_are_search_terms(self):
        for node in [
            NumberTerm(1, 1, False),
            TextTerm("x"),
            BytesTerm(b"\x00"),
            FloatTerm(1.0, 4),
            RangeTerm(0, 10),
        ]:
            assert isinstance(node, SearchTerm)


# ===================================================================
# Frontend tests
# ===================================================================


class TestSplitTextAndType:
    def test_hex_alias(self):
        value, spec = split_text_and_type("453f,h")
        assert value == "453f"
        assert spec is HexBytesSpec

    def test_no_comma_is_magic(self):
        value, spec = split_text_and_type("hello world")
        assert value == "hello world"
        assert spec is MagicSpec

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown type specifier"):
            split_text_and_type("123,ZZZZZ")

    def test_i32_alias(self):
        value, spec = split_text_and_type("0x100,i32")
        assert value == "0x100"
        assert spec is SignedIntSpec

    def test_float_alias(self):
        value, spec = split_text_and_type("2.5,lf")
        assert value == "2.5"
        assert spec is DoubleSpec

    def test_address_alias(self):
        value, spec = split_text_and_type("401000,addr")
        assert value == "401000"
        assert spec is AddressSpec

    def test_empty_alias_is_magic(self):
        value, spec = split_text_and_type("foo,")
        assert value == "foo"
        assert spec is MagicSpec


class TestHexBytesSpec:
    def test_plain_hex(self):
        [term] = HexBytesSpec.parse("453f")
        assert isinstance(term, BytesTerm)
        assert term.data == b"\x45\x3f"
        assert term.mask is None

    def test_hex_with_spaces(self):
        [term] = HexBytesSpec.parse("45 3f de ad")
        assert term.data == b"\x45\x3f\xde\xad"

    def test_wildcard(self):
        [term] = HexBytesSpec.parse("45 ?? 3f")
        assert term.data == b"\x45\x00\x3f"
        assert term.mask == b"\xff\x00\xff"

    def test_0x_prefix(self):
        [term] = HexBytesSpec.parse("0x41 0x42")
        assert term.data == b"\x41\x42"

    def test_semicolons_and_braces(self):
        [term] = HexBytesSpec.parse("{41;42}")
        assert term.data == b"\x41\x42"


class TestMaskedBytesSpec:
    def test_binary_mask(self):
        [term] = MaskedBytesSpec.parse("0b1?000001")
        assert isinstance(term, BytesTerm)
        assert term.data in (b"\x81", b"\xc1")
        assert term.mask is not None
        assert term.strict_mask is True

    def test_explicit_byte_mask_pair(self):
        [term] = MaskedBytesSpec.parse("41/DF")
        assert isinstance(term, BytesTerm)
        assert term.mask is not None

    def test_all_exact_no_mask(self):
        [term] = MaskedBytesSpec.parse("41 42")
        assert term.data == b"\x41\x42"
        assert term.mask is None

    def test_hex_question_mark(self):
        [term] = MaskedBytesSpec.parse("4?")
        assert isinstance(term, BytesTerm)
        assert term.mask is not None


class TestIntegerSpecs:
    def test_signed_int_hex(self):
        [term] = SignedIntSpec.parse("0x100")
        assert isinstance(term, NumberTerm)
        assert term.value == 256
        assert term.width == 4
        assert term.signed is True

    def test_unsigned_int_decimal(self):
        [term] = UnsignedIntSpec.parse("42")
        assert term.value == 42
        assert term.width == 4
        assert term.signed is False

    def test_signed_byte_negative(self):
        [term] = SignedByteSpec.parse("-1")
        assert term.value == -1
        assert term.width == 1
        assert term.signed is True

    def test_unsigned_byte(self):
        [term] = UnsignedByteSpec.parse("0xFF")
        assert term.value == 255
        assert term.width == 1

    def test_signed_short(self):
        [term] = SignedShortSpec.parse("0o777")
        assert term.value == 0o777
        assert term.width == 2

    def test_unsigned_short(self):
        [term] = UnsignedShortSpec.parse("0b1010")
        assert term.value == 0b1010
        assert term.width == 2

    def test_quad(self):
        [term] = SignedQuadSpec.parse("0x1000000000")
        assert term.value == 0x1000000000
        assert term.width == 8

    def test_unsigned_quad(self):
        [term] = UnsignedQuadSpec.parse("123456789")
        assert term.value == 123456789
        assert term.width == 8

    def test_address_hex_without_prefix(self):
        [term] = AddressSpec.parse("401000")
        assert term.value == 0x401000
        assert term.width == 3
        assert term.signed is False

    def test_address_hex_with_prefix(self):
        [term] = AddressSpec.parse("0x401000")
        assert term.value == 0x401000
        assert term.width == 3


class TestTextSpecs:
    def test_text_spec(self):
        [term] = TextSpec.parse("hello")
        assert isinstance(term, TextTerm)
        assert term.text == "hello"
        assert term.encoding is None

    def test_ascii_spec(self):
        [term] = ASCIISpec.parse("abc")
        assert term.encoding == "ascii+latin1"

    def test_unicode_spec(self):
        [term] = UnicodeSpec.parse("test")
        assert term.encoding == "unicode"

    def test_ebcdic_spec(self):
        [term] = EBCDICSpec.parse("data")
        assert term.encoding == "cp500"


class TestFloatSpecs:
    def test_float_spec(self):
        [term] = FloatSpec.parse("2.5")
        assert isinstance(term, FloatTerm)
        assert term.value == 2.5
        assert term.width == 4

    def test_double_spec(self):
        [term] = DoubleSpec.parse("3.14")
        assert term.value == pytest.approx(3.14)
        assert term.width == 8


class TestRangeSpec:
    def test_decimal_range(self):
        [term] = RangeSpec.parse("915..919")
        assert isinstance(term, RangeTerm)
        assert term.low == 915
        assert term.high == 919

    def test_hex_range(self):
        [term] = RangeSpec.parse("0x100..0x200")
        assert isinstance(term, RangeTerm)
        assert term.low == 0x100
        assert term.high == 0x200

    def test_mixed_bases(self):
        [term] = RangeSpec.parse("0o10..0xFF")
        assert term.low == 8
        assert term.high == 255

    def test_single_value_range(self):
        [term] = RangeSpec.parse("42..42")
        assert term.low == 42
        assert term.high == 42

    def test_missing_dots_raises(self):
        with pytest.raises(ValueError, match="low..high"):
            RangeSpec.parse("100")

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="must be <="):
            RangeSpec.parse("200..100")

    def test_split_text_and_type_range(self):
        value, spec = split_text_and_type("915..919,range")
        assert value == "915..919"
        assert spec is RangeSpec

    def test_split_text_and_type_r(self):
        value, spec = split_text_and_type("0x100..0x200,r")
        assert value == "0x100..0x200"
        assert spec is RangeSpec

    def test_alias_registered(self):
        assert AliasToType["range"] is RangeSpec
        assert AliasToType["r"] is RangeSpec


class TestMagicSpec:
    def test_returns_multiple_terms(self):
        terms = MagicSpec.parse("42")
        assert len(terms) > 1
        has_number = any(isinstance(t, NumberTerm) for t in terms)
        assert has_number

    def test_text_input_doesnt_crash(self):
        terms = MagicSpec.parse("hello world")
        assert isinstance(terms, list)
        has_text = any(isinstance(t, TextTerm) for t in terms)
        assert has_text

    def test_hex_like_input(self):
        terms = MagicSpec.parse("DEADBEEF")
        assert len(terms) >= 1

    def test_magic_detects_range_syntax(self):
        terms = MagicSpec.parse("100..200")
        has_range = any(isinstance(t, RangeTerm) for t in terms)
        assert has_range


# ===================================================================
# Backend emit tests (no IDA)
# ===================================================================


class TestOverflowSafety:
    """Values too large for the requested width must not crash."""

    backend = ByteSearchBackend()

    def test_value_too_big_for_1_byte(self):
        term = NumberTerm(value=1010010, width=1, signed=False)
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=True)
        assert patterns == []

    def test_value_too_big_for_2_bytes(self):
        term = NumberTerm(value=1010010, width=2, signed=True)
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=True)
        assert patterns == []

    def test_value_too_big_for_1_byte_big_endian(self):
        term = NumberTerm(value=0xFFFF, width=1, signed=False)
        patterns = self.backend.emit(term, Endian.BIG, [], case_sensitive=True)
        assert patterns == []

    def test_value_too_big_for_1_byte_both_endian(self):
        term = NumberTerm(value=0x1234, width=1, signed=False)
        patterns = self.backend.emit(term, Endian.BOTH, [], case_sensitive=True)
        assert patterns == []

    def test_negative_overflow_unsigned(self):
        term = NumberTerm(value=-1, width=1, signed=False)
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=True)
        assert patterns == []

    def test_value_fits_produces_pattern(self):
        term = NumberTerm(value=255, width=1, signed=False)
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=True)
        assert len(patterns) == 1
        assert patterns[0].data == b"\xff"

    def test_magic_large_value_no_crash(self):
        loc = PatternLocator.from_string("1010010", case_sensitive=True)
        patterns = loc.to_pattern()
        assert isinstance(patterns, list)
        fitting = [p for p in patterns if len(p.data) >= 4]
        assert len(fitting) >= 1

    def test_magic_very_large_value_no_crash(self):
        loc = PatternLocator.from_string("999999999999999999999", case_sensitive=True)
        patterns = loc.to_pattern()
        assert isinstance(patterns, list)


class TestByteSearchBackend:
    backend = ByteSearchBackend()

    def test_number_little_endian(self):
        term = NumberTerm(value=0x100, width=4, signed=True)
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        assert len(patterns) == 1
        assert patterns[0].data == (0x100).to_bytes(4, "little", signed=True)

    def test_number_big_endian(self):
        term = NumberTerm(value=0x100, width=4, signed=False)
        patterns = self.backend.emit(term, Endian.BIG, [])
        assert len(patterns) == 1
        assert patterns[0].data == (0x100).to_bytes(4, "big")

    def test_number_both_endian(self):
        term = NumberTerm(value=0x100, width=4, signed=False)
        patterns = self.backend.emit(term, Endian.BOTH, [])
        le = (0x100).to_bytes(4, "little")
        be = (0x100).to_bytes(4, "big")
        data_set = {p.data for p in patterns}
        assert le in data_set
        assert be in data_set

    def test_text_default_encodings(self):
        term = TextTerm(text="AB")
        patterns = self.backend.emit(term, Endian.LITTLE, ["utf-8", "utf-16le"])
        data_set = {p.data for p in patterns}
        assert b"AB" in data_set
        assert "AB".encode("utf-16le") in data_set

    def test_text_ascii_latin1(self):
        term = TextTerm(text="A", encoding="ascii+latin1")
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        data_set = {p.data for p in patterns}
        assert b"A" in data_set

    def test_text_unicode_little(self):
        term = TextTerm(text="X", encoding="unicode")
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        data_set = {p.data for p in patterns}
        assert b"X" in data_set  # utf-8
        assert "X".encode("utf-16le") in data_set
        assert "X".encode("utf-32le") in data_set

    def test_text_unicode_big(self):
        term = TextTerm(text="X", encoding="unicode")
        patterns = self.backend.emit(term, Endian.BIG, [])
        data_set = {p.data for p in patterns}
        assert "X".encode("utf-16be") in data_set
        assert "X".encode("utf-32be") in data_set

    def test_text_unicode_both(self):
        term = TextTerm(text="A", encoding="unicode")
        patterns = self.backend.emit(term, Endian.BOTH, [])
        data_set = {p.data for p in patterns}
        assert "A".encode("utf-16le") in data_set
        assert "A".encode("utf-16be") in data_set

    def test_text_cp500(self):
        term = TextTerm(text="A", encoding="cp500")
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=True)
        assert len(patterns) == 1
        assert patterns[0].data == "A".encode("cp500")

    def test_text_cp500_case_insensitive(self):
        term = TextTerm(text="A", encoding="cp500")
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        data_set = {p.data for p in patterns}
        assert "A".encode("cp500") in data_set
        assert "a".encode("cp500") in data_set

    def test_bytes_passthrough(self):
        term = BytesTerm(data=b"\xde\xad", mask=b"\xff\xff")
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        assert len(patterns) == 1
        assert patterns[0].data == b"\xde\xad"
        assert patterns[0].mask == b"\xff\xff"

    def test_bytes_no_mask(self):
        term = BytesTerm(data=b"\x90\x90")
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        assert patterns[0].mask is None

    def test_float_little(self):
        term = FloatTerm(value=2.5, width=4)
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        assert len(patterns) == 1
        assert patterns[0].data == struct.pack("<f", 2.5)

    def test_float_big(self):
        term = FloatTerm(value=2.5, width=4)
        patterns = self.backend.emit(term, Endian.BIG, [])
        assert patterns[0].data == struct.pack(">f", 2.5)

    def test_float_both(self):
        term = FloatTerm(value=1.0, width=8)
        patterns = self.backend.emit(term, Endian.BOTH, [])
        data_set = {p.data for p in patterns}
        assert struct.pack("<d", 1.0) in data_set
        assert struct.pack(">d", 1.0) in data_set


class TestByteSearchBackendRange:
    backend = ByteSearchBackend()

    def test_range_returns_empty(self):
        term = RangeTerm(low=100, high=200)
        patterns = self.backend.emit(term, Endian.LITTLE, [])
        assert patterns == []


class TestInsnOperandBackend:
    backend = InsnOperandBackend()

    def test_number_produces_query(self):
        term = NumberTerm(value=0x1234, width=4, signed=False)
        q = self.backend.emit(term)
        assert isinstance(q, OperandQuery)
        assert 0x1234 in q.values

    def test_text_returns_none(self):
        assert self.backend.emit(TextTerm(text="x")) is None

    def test_bytes_returns_none(self):
        assert self.backend.emit(BytesTerm(data=b"\x00")) is None

    def test_float_returns_none(self):
        assert self.backend.emit(FloatTerm(value=1.0, width=4)) is None

    def test_range_produces_query(self):
        term = RangeTerm(low=100, high=200)
        q = self.backend.emit(term)
        assert isinstance(q, OperandQuery)
        assert q.ranges == ((100, 200),)
        assert q.values == frozenset()

    def test_operand_query_matches_in_range(self):
        q = OperandQuery(values=frozenset(), ranges=((10, 20),))
        assert q.matches(10)
        assert q.matches(15)
        assert q.matches(20)
        assert not q.matches(9)
        assert not q.matches(21)

    def test_operand_query_matches_exact_values(self):
        q = OperandQuery(values=frozenset({42}), ranges=())
        assert q.matches(42)
        assert not q.matches(43)

    def test_operand_query_matches_both(self):
        q = OperandQuery(values=frozenset({5}), ranges=((10, 20),))
        assert q.matches(5)
        assert q.matches(15)
        assert not q.matches(7)


class TestMicrocodeBackend:
    backend = MicrocodeBackend()

    def test_number_produces_query(self):
        term = NumberTerm(value=42, width=4, signed=True)
        q = self.backend.emit(term)
        assert isinstance(q, MicrocodeQuery)
        assert 42 in q.values

    def test_text_produces_query(self):
        term = TextTerm(text="hello")
        q = self.backend.emit(term)
        assert isinstance(q, MicrocodeQuery)
        assert q.text == "hello"

    def test_float_produces_query(self):
        term = FloatTerm(value=1.0, width=4)
        q = self.backend.emit(term)
        assert isinstance(q, MicrocodeQuery)
        assert q.float_value == pytest.approx(1.0)

    def test_bytes_returns_none(self):
        assert self.backend.emit(BytesTerm(data=b"\x00")) is None

    def test_range_produces_query(self):
        term = RangeTerm(low=915, high=919)
        q = self.backend.emit(term)
        assert isinstance(q, MicrocodeQuery)
        assert q.ranges == [(915, 920)]
        assert q.values == []


class TestCTreeBackend:
    backend = CTreeBackend()

    def test_number_produces_query(self):
        term = NumberTerm(value=0xFF, width=4, signed=False)
        q = self.backend.emit(term)
        assert isinstance(q, CTreeQuery)
        assert q.number == 0xFF

    def test_text_produces_query(self):
        term = TextTerm(text="abc")
        q = self.backend.emit(term)
        assert isinstance(q, CTreeQuery)
        assert q.text == "abc"

    def test_float_produces_query(self):
        term = FloatTerm(value=3.14, width=8)
        q = self.backend.emit(term)
        assert isinstance(q, CTreeQuery)
        assert q.float_value == pytest.approx(3.14)
        assert q.cmat is None

    def test_bytes_returns_none(self):
        assert self.backend.emit(BytesTerm(data=b"\x00")) is None

    def test_range_produces_query(self):
        term = RangeTerm(low=0x100, high=0x200)
        q = self.backend.emit(term)
        assert isinstance(q, CTreeQuery)
        assert q.number_range == (0x100, 0x200)
        assert q.number is None

    def test_ctree_query_number_matches_exact(self):
        q = CTreeQuery(number=42)
        assert q.number_matches(42)
        assert not q.number_matches(43)
        assert q.has_number_query

    def test_ctree_query_number_matches_range(self):
        q = CTreeQuery(number_range=(10, 20))
        assert q.number_matches(10)
        assert q.number_matches(15)
        assert q.number_matches(20)
        assert not q.number_matches(9)
        assert not q.number_matches(21)
        assert q.has_number_query

    def test_ctree_query_no_number_query(self):
        q = CTreeQuery(text="hello")
        assert not q.has_number_query
        assert not q.number_matches(42)

    def test_iter_switch_case_values(self):
        class FakeCase:
            def __init__(self, values):
                self.values = values

        class FakeSwitch:
            def __init__(self, cases):
                self.cases = cases

        switch = FakeSwitch([FakeCase([1, 2]), FakeCase([7]), FakeCase([])])

        assert list(_iter_switch_case_values(switch)) == [1, 2, 7]

    def test_iter_switch_case_values_handles_none(self):
        assert list(_iter_switch_case_values(None)) == []

    def test_iter_ctree_numeric_values_reads_num_value(self):
        class FakeNumber:
            def __init__(self, value):
                self._value = value

        class FakeExpr:
            def __init__(self):
                self.op = object()
                self.n = FakeNumber(0x848)
                self.obj_ea = None

            def numval(self):
                raise AssertionError("fallback should not be needed")

        assert list(_iter_ctree_numeric_values(FakeExpr())) == [0x848]

    def test_iter_ctree_numeric_values_falls_back_to_numval(self):
        class FakeExpr:
            def __init__(self):
                self.op = object()
                self.n = None
                self.obj_ea = None

            def numval(self):
                return 0x848

        assert list(_iter_ctree_numeric_values(FakeExpr())) == [0x848]


class TestPseudocodeTextBackend:
    backend = PseudocodeTextBackend()

    def test_number_produces_hex_substring(self):
        term = NumberTerm(value=0x100, width=4, signed=False)
        q = self.backend.emit(term)
        assert isinstance(q, PseudocodeQuery)
        assert "100" in q.substring

    def test_text_produces_substring(self):
        term = TextTerm(text="foo_bar")
        q = self.backend.emit(term)
        assert q.substring == "foo_bar"

    def test_float_produces_str(self):
        term = FloatTerm(value=2.5, width=4)
        q = self.backend.emit(term)
        assert "2.5" in q.substring

    def test_bytes_returns_none(self):
        assert self.backend.emit(BytesTerm(data=b"\x00")) is None

    def test_range_returns_none(self):
        term = RangeTerm(low=100, high=200)
        assert self.backend.emit(term) is None


# ===================================================================
# Orchestration tests
# ===================================================================


class TestPatternLocator:
    def test_from_string_i32(self):
        loc = PatternLocator.from_string("0x100,i32", endian="little")
        assert loc.type_spec is SignedIntSpec
        assert loc.endian == Endian.LITTLE
        assert loc.value == "0x100"

    def test_from_string_default_encodings(self):
        loc = PatternLocator.from_string("hello,t")
        assert "utf-8" in loc.encodings

    def test_from_string_custom_encodings(self):
        loc = PatternLocator.from_string("x,t", encodings=["ascii"])
        assert loc.encodings == ("ascii",)

    def test_to_ir_number(self):
        loc = PatternLocator.from_string("256,u32")
        terms = loc.to_ir()
        assert len(terms) == 1
        assert isinstance(terms[0], NumberTerm)
        assert terms[0].value == 256

    def test_to_ir_text(self):
        loc = PatternLocator.from_string("hello,t")
        terms = loc.to_ir()
        assert len(terms) == 1
        assert isinstance(terms[0], TextTerm)

    def test_to_pattern_number(self):
        loc = PatternLocator.from_string("0x100,i32", endian="little")
        patterns = loc.to_pattern()
        assert len(patterns) >= 1
        expected = (0x100).to_bytes(4, "little", signed=True)
        assert any(p.data == expected for p in patterns)

    def test_to_pattern_hex_bytes(self):
        loc = PatternLocator.from_string("DEADBEEF,h")
        patterns = loc.to_pattern()
        assert len(patterns) == 1
        assert patterns[0].data == b"\xde\xad\xbe\xef"

    def test_to_pattern_float(self):
        loc = PatternLocator.from_string("2.5,f", endian="little")
        patterns = loc.to_pattern()
        assert any(p.data == struct.pack("<f", 2.5) for p in patterns)

    def test_to_operand_query_number(self):
        loc = PatternLocator.from_string("0x100,u32")
        q = loc.to_operand_query()
        assert isinstance(q, OperandQuery)
        assert 0x100 in q.values

    def test_to_operand_query_text_is_none(self):
        loc = PatternLocator.from_string("hello,t")
        assert loc.to_operand_query() is None

    def test_to_microcode_query_number(self):
        loc = PatternLocator.from_string("42,i32")
        q = loc.to_microcode_query()
        assert isinstance(q, MicrocodeQuery)
        assert 42 in q.values

    def test_to_microcode_query_text(self):
        loc = PatternLocator.from_string("hello,t")
        q = loc.to_microcode_query()
        assert q.text == "hello"

    def test_to_microcode_query_float(self):
        loc = PatternLocator.from_string("3.14,f")
        q = loc.to_microcode_query()
        assert isinstance(q, MicrocodeQuery)
        assert q.float_value == pytest.approx(3.14)

    def test_to_ctree_query_number(self):
        loc = PatternLocator.from_string("0xFF,u8")
        q = loc.to_ctree_query()
        assert isinstance(q, CTreeQuery)
        assert q.number == 0xFF

    def test_to_ctree_query_text(self):
        loc = PatternLocator.from_string("func_name,a")
        q = loc.to_ctree_query()
        assert q.text == "func_name"

    def test_to_ctree_query_float(self):
        loc = PatternLocator.from_string("3.14,f")
        q = loc.to_ctree_query()
        assert q.float_value == pytest.approx(3.14)

    def test_to_ctree_query_propagates_cmat(self):
        loc = PatternLocator.from_string("0xFF,u8")
        q = loc.to_ctree_query(cmat=7)
        assert isinstance(q, CTreeQuery)
        assert q.number == 0xFF
        assert q.cmat == 7

    def test_to_pseudocode_query_number(self):
        loc = PatternLocator.from_string("0x100,u32")
        q = loc.to_pseudocode_query()
        assert isinstance(q, PseudocodeQuery)
        assert "100" in q.substring

    def test_to_pseudocode_query_text(self):
        loc = PatternLocator.from_string("myvar,t")
        q = loc.to_pseudocode_query()
        assert q.substring == "myvar"

    def test_to_pseudocode_query_hex_bytes_is_none(self):
        loc = PatternLocator.from_string("DEAD,h")
        assert loc.to_pseudocode_query() is None

    def test_range_to_ir(self):
        loc = PatternLocator.from_string("915..919,range")
        terms = loc.to_ir()
        assert len(terms) == 1
        assert isinstance(terms[0], RangeTerm)
        assert terms[0].low == 915
        assert terms[0].high == 919

    def test_range_to_pattern_empty(self):
        loc = PatternLocator.from_string("100..200,range")
        patterns = loc.to_pattern()
        assert patterns == []

    def test_range_to_operand_query(self):
        loc = PatternLocator.from_string("100..200,range")
        q = loc.to_operand_query()
        assert isinstance(q, OperandQuery)
        assert q.ranges == ((100, 200),)

    def test_range_to_microcode_query(self):
        loc = PatternLocator.from_string("915..919,range")
        q = loc.to_microcode_query()
        assert isinstance(q, MicrocodeQuery)
        assert q.ranges == [(915, 920)]

    def test_range_to_microcode_query_with_reqmat(self):
        loc = PatternLocator.from_string("915..919,range")
        q = loc.to_microcode_query(reqmat=5)
        assert isinstance(q, MicrocodeQuery)
        assert q.ranges == [(915, 920)]
        assert q.reqmat == 5

    def test_range_to_ctree_query(self):
        loc = PatternLocator.from_string("0x100..0x200,range")
        q = loc.to_ctree_query()
        assert isinstance(q, CTreeQuery)
        assert q.number_range == (0x100, 0x200)

    def test_range_to_ctree_query_with_cmat(self):
        loc = PatternLocator.from_string("100..200,range")
        q = loc.to_ctree_query(cmat=7)
        assert isinstance(q, CTreeQuery)
        assert q.number_range == (100, 200)
        assert q.cmat == 7

    def test_magic_range_to_operand_query(self):
        loc = PatternLocator.from_string("0x848..0x849")
        q = loc.to_operand_query()
        assert isinstance(q, OperandQuery)
        assert q.ranges == ((0x848, 0x849),)

    def test_magic_range_to_microcode_query(self):
        loc = PatternLocator.from_string("0x848..0x849")
        q = loc.to_microcode_query()
        assert isinstance(q, MicrocodeQuery)
        assert q.ranges == [(0x848, 0x84A)]
        assert q.text is None

    def test_magic_range_to_ctree_query(self):
        loc = PatternLocator.from_string("0x848..0x849")
        q = loc.to_ctree_query()
        assert isinstance(q, CTreeQuery)
        assert q.number_range == (0x848, 0x849)
        assert q.text is None

    def test_magic_number_prefers_numeric_ctree_query(self):
        loc = PatternLocator.from_string("42")
        q = loc.to_ctree_query()
        assert isinstance(q, CTreeQuery)
        assert q.number == 42

    def test_range_to_pseudocode_query_is_none(self):
        loc = PatternLocator.from_string("100..200,range")
        assert loc.to_pseudocode_query() is None

    def test_to_string_roundtrip(self):
        loc = PatternLocator.from_string("0x100,i32")
        assert loc.to_string() == "0x100,i32"

    def test_equality(self):
        a = PatternLocator.from_string("0x100,i32", endian="little")
        b = PatternLocator.from_string("0x100,i32", endian="little")
        assert a == b

    def test_hash(self):
        a = PatternLocator.from_string("0x100,i32")
        b = PatternLocator.from_string("0x100,i32")
        assert hash(a) == hash(b)


class TestCaseInsensitive:
    """Tests for case-insensitive search across the pipeline."""

    backend = ByteSearchBackend()

    # -- ByteSearchBackend bitmask trick --

    def test_ascii_text_produces_masked_pattern(self):
        term = TextTerm(text="Ab", encoding="ascii+latin1")
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=False)
        masked = [p for p in patterns if p.mask is not None]
        assert len(masked) >= 1
        p = masked[0]
        assert p.strict_mask is True
        for i, b in enumerate(p.data):
            if 0x41 <= (b & 0xDF) <= 0x5A:
                assert p.mask[i] == 0xDF
            else:
                assert p.mask[i] == 0xFF

    def test_mask_values_for_hello(self):
        term = TextTerm(text="Hello1")
        patterns = self.backend.emit(
            term, Endian.LITTLE, ["utf-8"], case_sensitive=False
        )
        assert len(patterns) == 1
        p = patterns[0]
        assert p.mask is not None
        assert p.strict_mask is True
        assert p.mask == bytes([0xDF, 0xDF, 0xDF, 0xDF, 0xDF, 0xFF])

    def test_no_alpha_no_mask(self):
        term = TextTerm(text="123!")
        patterns = self.backend.emit(
            term, Endian.LITTLE, ["utf-8"], case_sensitive=False
        )
        assert len(patterns) == 1
        assert patterns[0].mask is None

    def test_case_sensitive_no_mask(self):
        term = TextTerm(text="Hello")
        patterns = self.backend.emit(
            term, Endian.LITTLE, ["utf-8"], case_sensitive=True
        )
        assert len(patterns) == 1
        assert patterns[0].mask is None
        assert patterns[0].data == b"Hello"

    def test_utf16le_mask(self):
        term = TextTerm(text="A", encoding="unicode")
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=False)
        utf16le_pats = [p for p in patterns if len(p.data) == 2]
        assert len(utf16le_pats) >= 1
        p = utf16le_pats[0]
        assert p.data == b"\x41\x00"
        assert p.mask == bytes([0xDF, 0xFF])
        assert p.strict_mask is True

    def test_utf16be_mask(self):
        term = TextTerm(text="A", encoding="unicode")
        patterns = self.backend.emit(term, Endian.BIG, [], case_sensitive=False)
        utf16be_pats = [p for p in patterns if len(p.data) == 2]
        assert len(utf16be_pats) >= 1
        p = utf16be_pats[0]
        assert p.data == b"\x00\x41"
        assert p.mask == bytes([0xFF, 0xDF])

    def test_cp500_fallback_variants(self):
        term = TextTerm(text="Ab", encoding="cp500")
        patterns = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=False)
        data_set = {p.data for p in patterns}
        assert "AB".encode("cp500") in data_set
        assert "ab".encode("cp500") in data_set

    def test_number_not_affected(self):
        term = NumberTerm(value=0x41, width=1, signed=False)
        pats_ci = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=False)
        pats_cs = self.backend.emit(term, Endian.LITTLE, [], case_sensitive=True)
        assert pats_ci == pats_cs

    # -- Microcode backend propagation --

    def test_microcode_emit_propagates_case_sensitive(self):
        mb = MicrocodeBackend()
        q = mb.emit(TextTerm(text="Foo"), case_sensitive=False)
        assert q is not None
        assert q.case_sensitive is False
        q2 = mb.emit(TextTerm(text="Foo"), case_sensitive=True)
        assert q2.case_sensitive is True

    def test_microcode_number_ignores_case(self):
        mb = MicrocodeBackend()
        q = mb.emit(NumberTerm(value=42, width=4, signed=True), case_sensitive=False)
        assert q is not None
        assert q.case_sensitive is False

    # -- CTree backend propagation --

    def test_ctree_emit_propagates_case_sensitive(self):
        cb = CTreeBackend()
        q = cb.emit(TextTerm(text="Bar"), case_sensitive=False)
        assert q is not None
        assert q.case_sensitive is False
        q2 = cb.emit(TextTerm(text="Bar"), case_sensitive=True)
        assert q2.case_sensitive is True

    # -- Pseudocode backend propagation --

    def test_pseudocode_emit_propagates_case_sensitive(self):
        pb = PseudocodeTextBackend()
        q = pb.emit(TextTerm(text="Baz"), case_sensitive=False)
        assert q is not None
        assert q.case_sensitive is False
        q2 = pb.emit(TextTerm(text="Baz"), case_sensitive=True)
        assert q2.case_sensitive is True

    def test_pseudocode_number_default_case(self):
        pb = PseudocodeTextBackend()
        q = pb.emit(NumberTerm(value=0xFF, width=4, signed=False))
        assert q.case_sensitive is False

    # -- PatternLocator propagation --

    def test_locator_default_case_insensitive(self):
        loc = PatternLocator.from_string("hello,t")
        assert loc.case_sensitive is False

    def test_locator_explicit_case_sensitive(self):
        loc = PatternLocator.from_string("hello,t", case_sensitive=True)
        assert loc.case_sensitive is True

    def test_locator_to_pattern_case_insensitive(self):
        loc = PatternLocator.from_string("Ab,a", case_sensitive=False)
        patterns = loc.to_pattern()
        masked = [p for p in patterns if p.mask is not None]
        assert len(masked) >= 1
        assert all(p.strict_mask for p in masked)

    def test_locator_to_pattern_case_sensitive(self):
        loc = PatternLocator.from_string("Ab,a", case_sensitive=True)
        patterns = loc.to_pattern()
        assert all(p.mask is None for p in patterns)

    def test_locator_to_microcode_query_propagates(self):
        loc = PatternLocator.from_string("hello,t", case_sensitive=False)
        q = loc.to_microcode_query()
        assert q is not None
        assert q.case_sensitive is False

    def test_locator_to_ctree_query_propagates(self):
        loc = PatternLocator.from_string("hello,t", case_sensitive=True)
        q = loc.to_ctree_query()
        assert q is not None
        assert q.case_sensitive is True

    def test_locator_to_pseudocode_query_propagates(self):
        loc = PatternLocator.from_string("hello,t", case_sensitive=False)
        q = loc.to_pseudocode_query()
        assert q is not None
        assert q.case_sensitive is False

    def test_locator_equality_respects_case_flag(self):
        a = PatternLocator.from_string("hello,t", case_sensitive=True)
        b = PatternLocator.from_string("hello,t", case_sensitive=False)
        assert a != b


class TestDisplayHelpers:
    def test_help_not_empty(self):
        h = help()
        assert len(h) > 100
        assert "hex" in h.lower() or "Hex" in h

    def test_overview_not_empty(self):
        o = overview()
        assert len(o) > 10

    def test_html_overview_contains_table(self):
        h = html_overview()
        assert "<table>" in h
        assert "</table>" in h

    def test_overview_contains_categories(self):
        o = overview()
        assert "text" in o.lower()


class TestRegistry:
    def test_types_is_not_empty(self):
        assert len(Types) > 5

    def test_alias_to_type_has_h(self):
        assert AliasToType["h"] is HexBytesSpec

    def test_alias_to_type_has_none(self):
        assert AliasToType[None] is MagicSpec

    def test_alias_to_type_has_i32(self):
        assert AliasToType["i32"] is SignedIntSpec

    def test_alias_to_type_has_addr(self):
        assert AliasToType["addr"] is AddressSpec
