"""
IDA plugin entry point for IDA Search.

Registers four actions:

    Alt-B           Quick byte search (ask_str dialog)
    Alt-Shift-B     Advanced search (form with backend checkboxes)
    Ctrl-B          Search next
    Ctrl-Shift-B    Search previous

The advanced search form lets the user select which backends to run:
byte patterns, instruction operands, microcode, ctree, pseudocode text.

Author: Milankovo, 2025
License: MIT
Inspired by: https://www.sweetscape.com/010editor/manual/Find.htm
"""

import json
import logging
from pathlib import Path
import typing

import idaapi

import parse
import ask_form
from ida_bytes import __to_bytevec

SELECTION_THRESHOLD = 64
PLUGIN_DISPLAY_NAME = "IDA Search"
PLUGIN_SLUG = "ida-search"
ACTION_NAMESPACE = "ids"


logger = logging.getLogger(PLUGIN_SLUG)


def _load_plugin_manifest() -> dict:
    manifest_path = Path(__file__).with_name("ida-plugin.json")
    with manifest_path.open("r", encoding="ascii") as handle:
        return json.load(handle)["plugin"]


PLUGIN_MANIFEST = _load_plugin_manifest()
PLUGIN_VERSION = PLUGIN_MANIFEST["version"]
PLUGIN_REPOSITORY_URL = PLUGIN_MANIFEST["urls"]["repository"]


class search_parameters_t:
    def __init__(self):
        self.start_ea = idaapi.BADADDR
        self.end_ea = idaapi.BADADDR

        self.byte_patterns: list[parse.Pattern] = []

    def __str__(self):
        return f"search_parameters_t(start={self.start_ea:#x}, end={self.end_ea:#x}, patterns={len(self.byte_patterns)})"

    def has_patterns(self) -> bool:
        return bool(self.byte_patterns)

    def search_next(self, start_ea: int):
        """
        Search for the next occurence of the pattern
        """
        if not self.has_patterns():
            return idaapi.BADADDR

        patterns, flags = self.compile_search_patterns()

        if self.start_ea <= start_ea < self.end_ea:
            ea, _ = idaapi.bin_search(
                start_ea, self.end_ea, patterns, idaapi.BIN_SEARCH_FORWARD | flags
            )
            return ea

        ea, _ = idaapi.bin_search(
            start_ea,
            idaapi.inf_get_max_ea(),
            patterns,
            idaapi.BIN_SEARCH_FORWARD | flags,
        )
        return ea

    def search_prev(self, start_ea: int):
        """
        Search for the previous occurence of the pattern
        """
        if not self.has_patterns():
            return idaapi.BADADDR

        patterns, flags = self.compile_search_patterns()

        if self.start_ea <= start_ea < self.end_ea:
            ea, _ = idaapi.bin_search(
                self.start_ea, start_ea, patterns, idaapi.BIN_SEARCH_BACKWARD | flags
            )
            return ea

        ea, _ = idaapi.bin_search(
            idaapi.inf_get_min_ea(),
            start_ea,
            patterns,
            idaapi.BIN_SEARCH_BACKWARD | flags,
        )
        return ea

    def reset_range(self):
        self.start_ea = idaapi.BADADDR
        self.end_ea = idaapi.BADADDR

    def set_range(self, start_ea: int, end_ea: int):
        self.start_ea = start_ea
        self.end_ea = end_ea

    def set_patterns(self, byte_patterns: list[parse.Pattern]):
        self.byte_patterns = byte_patterns

    def search_all(self) -> typing.Generator[tuple[int, int], None, None]:
        """
        Search for all occurences of the pattern
        """
        if not self.has_patterns():
            return

        patterns, flags = self.compile_search_patterns()

        from backend import wait_box

        with wait_box("Searching bytes..."):
            ea = self.start_ea
            while ea < self.end_ea:
                ea, idx = idaapi.bin_search(
                    ea, self.end_ea, patterns, flags | idaapi.BIN_SEARCH_FORWARD
                )
                if ea == idaapi.BADADDR:
                    break

                yield ea, len(patterns[idx].bytes)
                ea += 1
                if ea % 0x10000 == 0:
                    idaapi.show_addr(ea)
                if idaapi.user_cancelled():
                    return

    def compile_search_patterns(self):
        patterns = idaapi.compiled_binpat_vec_t()

        flags = 0
        for p in self.byte_patterns:
            add_pattern(patterns, p.data, p.mask)
            if p.mask is not None and p.strict_mask:
                flags |= idaapi.BIN_SEARCH_BITMASK
        return patterns, flags


class address_chooser(idaapi.Choose):
    """
    based on ida/python/examples/ui/tabular_views/custom/func_chooser.py
    """

    def __init__(self, title, eas: list[tuple[int, int]] | None = None):
        idaapi.Choose.__init__(
            self,
            title,
            [
                ["Address", 10 | idaapi.Choose.CHCOL_EA],
                [
                    "Function",
                    15 | idaapi.Choose.CHCOL_FNAME,
                ],
                ["Bytes", 15 | idaapi.Choose.CHCOL_HEX],
                ["Instruction", 30 | idaapi.Choose.CHCOL_PLAIN],
            ],
        )

        if eas is None:
            eas = []
        self.items = []
        self.eas = eas
        self.icon = idaapi.get_icon_id_by_name("resources/menu/OpenFunctions.svg")

    def OnInit(self):
        def dis(ea):
            return idaapi.generate_disasm_line(ea, idaapi.GENDSM_REMOVE_TAGS | idaapi.GENDSM_FORCE_CODE).strip()

        def nm(ea):
            return idaapi.get_func_name(ea) or idaapi.get_name(ea) or ""

        def get_hexbytes(ea, sz):
            return idaapi.get_bytes(ea, sz).hex(" ")

        self.items = [
            [hex(ea), nm(ea), get_hexbytes(ea, sz), dis(ea), ea] for ea, sz in self.eas
        ]
        return True

    def OnGetSize(self):
        return len(self.items)

    def OnGetLine(self, n):
        return self.items[n]

    def OnGetEA(self, n):
        return self.items[n][-1]

    def OnRefresh(self, n):
        self.OnInit()
        # try to preserve the cursor
        return [idaapi.Choose.ALL_CHANGED] + self.adjust_last_item(n)


def show_addresses(eas: list[tuple[int, int]], modal=False):
    c = address_chooser("Search results", eas)
    c.Show(modal=modal)


class result_chooser(idaapi.Choose):
    """Chooser for ``(ea, description)`` results (microcode, ctree, insn operand)."""

    def __init__(self, title, results: list[tuple[int, str]] | None = None):
        idaapi.Choose.__init__(
            self,
            title,
            [
                ["Address", 10 | idaapi.Choose.CHCOL_EA],
                ["Function", 15 | idaapi.Choose.CHCOL_FNAME],
                ["Description", 40 | idaapi.Choose.CHCOL_PLAIN],
            ],
        )
        self.results = results or []
        self.items = []
        self.icon = idaapi.get_icon_id_by_name("resources/menu/OpenFunctions.svg")

    def OnInit(self):
        def nm(ea):
            return idaapi.get_func_name(ea) or idaapi.get_name(ea) or ""

        self.items = [[hex(ea), nm(ea), desc, ea] for ea, desc in self.results]
        return True

    def OnGetSize(self):
        return len(self.items)

    def OnGetLine(self, n):
        return self.items[n]

    def OnGetEA(self, n):
        return self.items[n][-1]

    def OnRefresh(self, n):
        self.OnInit()
        return [idaapi.Choose.ALL_CHANGED] + self.adjust_last_item(n)


def show_results(title: str, results: list[tuple[int, str]], modal=False):
    c = result_chooser(title, results)
    c.Show(modal=modal)


class pseudocode_chooser(idaapi.Choose):
    """Chooser for ``(func_ea, line_text, line_no)`` pseudocode results."""

    def __init__(self, title, results: list[tuple[int, str, int]] | None = None):
        idaapi.Choose.__init__(
            self,
            title,
            [
                ["Function", 15 | idaapi.Choose.CHCOL_FNAME],
                ["Line", 5 | idaapi.Choose.CHCOL_DEC],
                ["Text", 60 | idaapi.Choose.CHCOL_PLAIN],
            ],
        )
        self.results = results or []
        self.items = []
        self.icon = idaapi.get_icon_id_by_name("resources/menu/OpenFunctions.svg")

    def OnInit(self):
        def nm(ea):
            return idaapi.get_func_name(ea) or idaapi.get_name(ea) or ""

        self.items = [
            [nm(func_ea), str(line_no), text.strip(), func_ea]
            for func_ea, text, line_no in self.results
        ]
        return True

    def OnGetSize(self):
        return len(self.items)

    def OnGetLine(self, n):
        return self.items[n]

    def OnGetEA(self, n):
        return self.items[n][-1]

    def OnRefresh(self, n):
        self.OnInit()
        return [idaapi.Choose.ALL_CHANGED] + self.adjust_last_item(n)


def show_pseudocode_results(
    title: str, results: list[tuple[int, str, int]], modal=False
):
    c = pseudocode_chooser(title, results)
    c.Show(modal=modal)


def test_search():
    # test search
    patterns = idaapi.compiled_binpat_vec_t()
    add_pattern(patterns, b"\x90\x90\x90", b"\xff\xff\xff")
    ea, _ = idaapi.bin_search(
        idaapi.inf_get_max_ea(),
        idaapi.inf_get_max_ea(),
        patterns,
        idaapi.BIN_SEARCH_FORWARD,
    )
    logger.info("test_search result: %s", hex(ea))


def test_parse_binpat_str():
    patterns = idaapi.compiled_binpat_vec_t()
    idaapi.parse_binpat_str(patterns, 0, "90 ? 90", 16, idaapi.BPU_1B)
    logger.info("compiled patterns: %s", patterns.size())
    for i in range(patterns.size()):
        logger.info("pattern[%s].bytes=%s", i, patterns[i].bytes)
        logger.info("pattern[%s].mask=%s", i, patterns[i].mask)


def add_pattern(
    patterns: idaapi.compiled_binpat_vec_t, data: bytes, mask: bytes | None = None
):
    p0 = patterns.push_back()
    p0.bytes = __to_bytevec(data)
    if mask is not None:
        p0.mask = __to_bytevec(mask)


def search_bytes(
    byte_pattern: bytes, start_ea: int | None = None, end_ea: int | None = None
):
    """
    Search for a byte pattern in the database
    """
    patterns = idaapi.compiled_binpat_vec_t()

    if start_ea is None:
        # lowest address
        start_ea = idaapi.inf_get_min_ea()
    if end_ea is None:
        # highest address
        end_ea = idaapi.inf_get_max_ea()

    # idaapi.parse_binpat_str(patterns, start_ea, s, 16, idaapi.BPU_1B)
    add_pattern(patterns, byte_pattern)

    ea = start_ea
    while ea < end_ea:
        ea, _ = idaapi.bin_search(ea, end_ea, patterns, idaapi.BIN_SEARCH_FORWARD)
        if ea == idaapi.BADADDR:
            break
        yield ea
        ea += 1


def val_to_pattern(val: int):
    bl = val.bit_length()
    if bl <= 8:
        return f"{val:#x},u8"
    if bl <= 16:
        return f"{val:#x},u16"
    if bl <= 32:
        return f"{val:#x},u32"
    if bl <= 64:
        return f"{val:#x},u64"
    if bl <= 128:
        return f"{val:#x},u128"
    if bl <= 256:
        return f"{val:#x},u256"
    return f"{val:#x},u512"


def build_default_query(ctx: idaapi.action_ctx_base_t, last_query: str = "") -> str:
    default_query = last_query

    current_highlight = idaapi.get_highlight(ctx.widget, 0)
    if current_highlight is not None:
        highlighted_value, _ = current_highlight
        try:
            val = int(highlighted_value, 16)
            if val == ctx.cur_ea:
                default_query = val_to_pattern(val)
        except ValueError:
            pass

    if default_query == last_query and ctx.cur_value != idaapi.BADADDR:
        default_query = val_to_pattern(ctx.cur_value)

    if ctx.has_flag(idaapi.ACF_HAS_SELECTION):
        ok, sel_start, sel_end = idaapi.read_range_selection(ctx.widget)
        if ok:
            selected_bytes = idaapi.get_bytes(sel_start, sel_end - sel_start)
            if selected_bytes is not None and len(selected_bytes) <= SELECTION_THRESHOLD:
                default_query = selected_bytes.hex(" ") + ",h"

    return default_query


class binary_search_ah_t(idaapi.action_handler_t):
    def __init__(self, search_parameters: search_parameters_t | None = None):
        super().__init__()
        self.last_query = ""
        self.search_parameters = search_parameters

    def activate(self, ctx: idaapi.action_ctx_base_t):
        start, end = idaapi.inf_get_min_ea(), idaapi.inf_get_max_ea()
        default_query = build_default_query(ctx, self.last_query)
        title = "Search for bytes"

        if ctx.has_flag(idaapi.ACF_HAS_SELECTION):
            ok, sel_start, sel_end = idaapi.read_range_selection(ctx.widget)
            if ok:
                selected_bytes = idaapi.get_bytes(sel_start, sel_end - sel_start)

                if selected_bytes is not None and len(selected_bytes) <= SELECTION_THRESHOLD:
                    default_query = selected_bytes.hex(" ") + ",h"
                else:
                    start, end = sel_start, sel_end
                    title = f"Search for bytes in {sel_start:x} - {sel_end:x}"

        q = idaapi.ask_str(default_query, idaapi.HIST_SRCH, title)
        if not q:
            return 0

        endian = "big" if idaapi.inf_is_be() else "little"
        b = parse.PatternLocator.from_string(q, endian=endian, case_sensitive=False)

        byte_patterns = b.to_pattern()

        if self.search_parameters is None:
            self.search_parameters = search_parameters_t()

        self.search_parameters.set_patterns(byte_patterns)
        self.search_parameters.set_range(start, end)

        eas = list(self.search_parameters.search_all())

        self.last_query = q
        if not eas:
            idaapi.info("Pattern not found")
            return 0

        show_addresses(eas, modal=False)
        return 1

    def update(self, ctx: idaapi.action_ctx_base_t):
        return idaapi.AST_ENABLE_ALWAYS


class advanced_search_ah_t(idaapi.action_handler_t):
    """Open the unified search form with backend checkboxes."""

    def __init__(self, search_parameters: search_parameters_t | None = None):
        super().__init__()
        self.last_query = ""
        self.last_backends_mask = 1
        self.last_options_mask = 0
        self.last_reqmat = ask_form.default_reqmat()
        self.last_cmat = ask_form.default_cmat()
        self.search_parameters = search_parameters

    def activate(self, ctx: idaapi.action_ctx_base_t):
        form = ask_form.binary_search_form(
            build_default_query(ctx, self.last_query),
            backends_mask=self.last_backends_mask,
            options_mask=self.last_options_mask,
            reqmat=self.last_reqmat,
            cmat=self.last_cmat,
        )
        form, args = form.Compile()
        ok = form.Execute()
        if ok != 1:
            form.Free()
            return 0

        pattern_str = form.get_pattern()
        endian = form.get_endian()
        backends = form.get_backends_mask()
        options_mask = form.get_options_mask()
        reqmat = form.get_reqmat()
        cmat = form.get_cmat()
        case_sensitive = form.get_case_sensitive()
        form.Free()

        if not pattern_str:
            return 0

        self.last_query = pattern_str
        self.last_backends_mask = backends
        self.last_options_mask = options_mask
        self.last_reqmat = reqmat
        self.last_cmat = cmat

        encodings = []
        for i in range(1, idaapi.get_encoding_qty()):
            encodings.append(idaapi.get_encoding_name(i))

        loc = parse.PatternLocator.from_string(
            pattern_str,
            endian=endian,
            encodings=encodings or None,
            case_sensitive=case_sensitive,
        )

        start_ea = idaapi.inf_get_min_ea()
        end_ea = idaapi.inf_get_max_ea()

        found_anything = False

        if backends & (1 << ask_form.BACKEND_BYTES):
            byte_patterns = loc.to_pattern()
            if byte_patterns:
                if self.search_parameters is None:
                    self.search_parameters = search_parameters_t()
                self.search_parameters.set_patterns(byte_patterns)
                self.search_parameters.set_range(start_ea, end_ea)
                eas = list(self.search_parameters.search_all())
                if eas:
                    found_anything = True
                    show_addresses(eas, modal=False)

        if backends & (1 << ask_form.BACKEND_INSN):
            oq = loc.to_operand_query()
            if oq is not None:
                eas = list(parse.search_insn_operands(oq, start_ea, end_ea))
                if eas:
                    found_anything = True
                    show_addresses(eas, modal=False)

        if backends & (1 << ask_form.BACKEND_MICRO):
            mq = loc.to_microcode_query(reqmat=reqmat)
            if mq is not None:
                results = list(parse.search_microcode(mq))
                if results:
                    found_anything = True
                    show_results("Microcode search", results, modal=False)

        if backends & (1 << ask_form.BACKEND_CTREE):
            cq = loc.to_ctree_query(cmat=cmat)
            if cq is not None:
                results = list(parse.search_ctree(cq))
                if results:
                    found_anything = True
                    show_results("CTree search", results, modal=False)

        if backends & (1 << ask_form.BACKEND_PSEUDO):
            pq = loc.to_pseudocode_query()
            if pq is not None:
                results = list(parse.search_pseudocode(pq))
                if results:
                    found_anything = True
                    show_pseudocode_results("Pseudocode search", results, modal=False)

        if not found_anything:
            idaapi.info("No results found")
            return 0

        return 1

    def update(self, ctx: idaapi.action_ctx_base_t):
        return idaapi.AST_ENABLE_ALWAYS


class binary_search_next_ah_t(idaapi.action_handler_t):
    def __init__(
        self, search_parameters: search_parameters_t | None = None, direction: int = 1
    ):
        super().__init__()
        self.search_parameters = search_parameters
        self.direction = direction

    def activate(self, ctx: idaapi.action_ctx_base_t):
        if self.search_parameters is None or not self.search_parameters.has_patterns():
            return 0

        match self.direction:
            case 1:
                ea = self.search_parameters.search_next(
                    idaapi.next_head(ctx.cur_ea, idaapi.inf_get_max_ea())
                )
            case -1:
                ea = self.search_parameters.search_prev(ctx.cur_ea)
            case _:
                raise ValueError("Invalid direction")
        if ea == idaapi.BADADDR:
            idaapi.info("Pattern not found")
            return 0

        idaapi.jumpto(ea)
        return 1

    def update(self, ctx):
        return idaapi.AST_ENABLE_ALWAYS


class BinarySearchPlugin(idaapi.plugin_t):
    """IDA plugin class for IDA Search.

    Registers four actions on ``init()``:

        - ``ids:quick`` (Alt-B) -- quick byte search via ``ask_str``.
        - ``ids:advanced`` (Alt-Shift-B) -- advanced search form
          with backend checkboxes.
        - ``ids:next`` (Ctrl-B) -- find next match.
        - ``ids:prev`` (Ctrl-Shift-B) -- find previous match.
    """

    flags = idaapi.PLUGIN_FIX | idaapi.PLUGIN_HIDE
    comment = "010 Editor-style type-aware binary search with multiple backends"
    help = "Type-aware search: Alt-B for quick byte search, Alt-Shift-B for advanced multi-backend search"
    wanted_name = PLUGIN_DISPLAY_NAME
    search_bytes_action_name = f"{ACTION_NAMESPACE}:quick"
    search_next_action_name = f"{ACTION_NAMESPACE}:next"
    search_prev_action_name = f"{ACTION_NAMESPACE}:prev"
    search_advanced_action_name = f"{ACTION_NAMESPACE}:advanced"
    wanted_hotkey = ""
    search_parameters_t = None

    def init(self):
        addon = idaapi.addon_info_t()
        addon.id = "milankovo.ida-search"
        addon.name = PLUGIN_DISPLAY_NAME
        addon.producer = "Milankovo"
        addon.url = PLUGIN_REPOSITORY_URL
        addon.version = PLUGIN_VERSION
        idaapi.register_addon(addon)

        self.search_parameters_t = search_parameters_t()

        search_bytes_action = idaapi.action_desc_t(
            self.search_bytes_action_name,
            "search bytes",
            binary_search_ah_t(self.search_parameters_t),
            "Alt-B",
        )
        idaapi.register_action(search_bytes_action)

        search_next_action = idaapi.action_desc_t(
            self.search_next_action_name,
            "search next",
            binary_search_next_ah_t(self.search_parameters_t, 1),
            "Ctrl-B",
        )
        idaapi.register_action(search_next_action)

        search_prev_action = idaapi.action_desc_t(
            self.search_prev_action_name,
            "search previous",
            binary_search_next_ah_t(self.search_parameters_t, -1),
            "Ctrl-Shift-B",
        )
        idaapi.register_action(search_prev_action)

        advanced_action = idaapi.action_desc_t(
            self.search_advanced_action_name,
            "advanced search",
            advanced_search_ah_t(self.search_parameters_t),
            "Alt-Shift-B",
        )
        idaapi.register_action(advanced_action)
        idaapi.attach_action_to_menu(
            "Search/", self.search_advanced_action_name, idaapi.SETMENU_APP
        )

        return idaapi.PLUGIN_KEEP

    def term(self):
        idaapi.unregister_action(self.search_bytes_action_name)
        idaapi.unregister_action(self.search_next_action_name)
        idaapi.unregister_action(self.search_prev_action_name)
        idaapi.unregister_action(self.search_advanced_action_name)

    def run(self, arg):
        pass


def PLUGIN_ENTRY():
    return BinarySearchPlugin()
