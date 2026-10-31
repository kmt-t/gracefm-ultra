#!/usr/bin/env python3
"""GraceFM の gracefm/tables.py から、受信側(Rust)に要る定数表を engine/gfm-core/src/tables.rs に書き出す。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gracefm"))
from gracefm import tables as T  # noqa: E402


def arr(name, vals, ty="i64"):
    return f"pub const {name}: [{ty}; {len(vals)}] = [{', '.join(str(int(v)) for v in vals)}];\n"


out = "// tools/gen_rs_tables.py が ../gracefm/gracefm/tables.py から作る (手で書き換えない)。\n"
out += arr("HANN128_Q15", T.HANN128_Q15)
for name in ("ALT_IIR_32K_Q28", "ALT_IIR_48K_Q28"):
    secs = getattr(T, name)
    out += f"pub const {name}: [[i64; 5]; {len(secs)}] = [{', '.join('[' + ', '.join(str(int(v)) for v in s) + ']' for s in secs)}];\n"
open(os.path.join(HERE, "..", "engine", "gfm-core", "src", "tables.rs"), "w").write(out)
print("wrote tables.rs")
