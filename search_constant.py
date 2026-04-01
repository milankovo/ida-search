import idaapi
import idautils

class result_t:
    def __init__(self, value: int, dstr: str, ea: int):
        self.value = value
        self.dstr = dstr
        self.ea = ea

    def __repr__(self):
        return f"result_t(value={self.value:#x}, dstr='{self.dstr}', ea={self.ea:#x})"
    

class constant_visitor_t(idaapi.mop_visitor_t):
    def __init__(self, range: idaapi.rangeset_t):
        super().__init__()
        self.range = range
        self.found_constants: list[result_t] = []

    def visit_mop(self, op: idaapi.mop_t, type: idaapi.tinfo_t, is_target: bool):
        if not op.is_constant():
            return 0

        ins: idaapi.insn_t = self.curins
        
        if self.range.contains(v := op.unsigned_value()):
            self.found_constants.append(result_t(v, op.dstr(), ins.ea))

        return 0  # continue visiting


def lookup_in_func(fnc_ea: int, range: idaapi.rangeset_t) -> list[result_t] | None:
    if not idaapi.init_hexrays_plugin():
        print("vds12: Hex-rays is not available.")
        return
    pfn = idaapi.get_func(fnc_ea)
    if not pfn:
        print("Please position the cursor within a function")
        return

    # generate microcode
    hf = idaapi.hexrays_failure_t()

    mbr = idaapi.mba_ranges_t(pfn)

    decomp_flags: int = idaapi.DECOMP_ALL_BLKS | idaapi.DECOMP_NO_WAIT  # type: ignore
    opts = idaapi.MMAT_LOCOPT
    mba: idaapi.mba_t

    mba = idaapi.gen_microcode(mbr, hf, None, decomp_flags, opts)  # type: ignore
    if not mba:
        print(f"No microcode generated for function at {fnc_ea:#x}.")
        return

    visitor = constant_visitor_t(range)

    mba.for_all_ops(visitor)

    if not visitor.found_constants:
        return

    print(f"Found constants in the function {fnc_ea:#x}:")
    for result in visitor.found_constants:
        print(f"Value: {result.value:#x}, Description: {result.dstr}, Address: {result.ea:#x}")
    return visitor.found_constants


class waitbox_context_manager_t:
    def __init__(self, message) -> None:
        self.message = message

    def __enter__(self):
        idaapi.show_wait_box(self.message)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        idaapi.hide_wait_box()
        ...

def update_cmt(ea: int, s: str, repeatable:bool):
    existing_cmt = idaapi.get_cmt(ea, repeatable)  # ensure comment exists
    if existing_cmt and s in existing_cmt:
        return
    idaapi.append_cmt(ea, s, repeatable)  # append comment if not already present

def main():
    ranges = idaapi.rangeset_t()
    #ranges.add(0x1018FD20, 0x1018FD54)
    #ranges.add(0x1018D030, 0x1018D510)
    #ranges.add(0x100152F0, 0x100152F0+1)
    #ranges.add(0x101BD98A, 0x101BD98A+1)
    ranges.add(1570196119, 1570196119+1)


    with waitbox_context_manager_t("Searching for constants in functions..."):
        for fnc_ea in idautils.Functions():
            # update the indicator arrow in the navigator
            idaapi.show_addr(fnc_ea)
            fnc = idaapi.get_func(fnc_ea)
            if not fnc:
                continue
            try:
                found = lookup_in_func(fnc_ea, ranges)
                if found:
                    for result in found:
                        print(f"Function {fnc_ea:#x} contains constant: {repr(result)}")
                        idaapi.remember_problem(idaapi.PR_ATTN, result.ea, f"Found constant {result.value:#x} at {result.ea:#x}: {result.dstr}")
                        update_cmt(result.ea, f"Found constant {result.value:#x}", False)
                        idaapi.add_dref(result.ea, result.value, idaapi.dr_O | idaapi.XREF_USER)
            except Exception as e:
                idaapi.warning(f"Error processing function at {fnc_ea:#x}: {e}")
                return

            # did user cancel?
            if idaapi.user_cancelled():
                break


if __name__ == "__main__":
    main()
