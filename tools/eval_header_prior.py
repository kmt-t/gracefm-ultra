#!/usr/bin/env python3
"""
ヘッダの事前知識つき復号（段 A、engine/gfm-rx の --prior）の効果: 主系の基本 CRC が通らないフレームの割合を、Es/N0 ごとに、
事前知識なし（none）、フレーム番号と直前の指数（temporal）、さらに副系のピークからの指数（full）で比べる。副系がある組と、ない組。

  .venv/bin/python tools/eval_header_prior.py [--seconds 6]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import rs  # noqa: E402
from gracefm_ultra.gfm import GRACEFM  # noqa: E402,F401
from gracefm.profiles import PROFILES  # noqa: E402

PAIRS = [("B200", "SUB100"), ("B200M", "SUB50M"), ("M700", "SUB100-48K")]
MODES = ("none", "temporal", "full")


def stats(main, frames_a, n, sub=None, frames_b=None, mode="full"):
    """gfm-rx の標準エラー出力から、主系の統計を読む"""
    with tempfile.TemporaryDirectory() as d:
        a, b, o = (os.path.join(d, v) for v in ("a.sym", "b.sym", "o.raw"))
        rs.write_sym(a, frames_a)
        cmd = [os.path.join(rs.BIN, "gfm-rx"), "--profile", main, "--frames-a", a, "--n", str(n), "--out", o, "--prior", mode]
        if sub:
            rs.write_sym(b, frames_b)
            cmd += ["--profile-b", sub, "--frames-b", b]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-300:])
    m = {k: int(v) for k, v in re.findall(r"(frames|lost|crc_fail|high_crc_fail|recovered) (\d+)", r.stderr)}
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--esn0", default="2,4,6,8,10")
    ap.add_argument("--kinds", default="awgn,rayleigh")
    a = ap.parse_args()
    for main_name, sub_name in PAIRS:
        p = PROFILES[main_name]
        x = rs.clip(p, a.seconds)
        print(f"\n{main_name} + {sub_name}（Es/N0 は主系・副系とも同じ）。主系の基本 CRC 不一致の割合 [%]\n")
        print(f"{'通信路':14}" + "".join(f"{m:>26}" for m in ("副系なし: none / temporal", "副系あり: none / temporal / full")))
        for kind in a.kinds.split(","):
            for e in [float(v) for v in a.esn0.split(",")]:
                fade = int(kind == "rayleigh")
                fa = rs.tx(main_name, x, scen=(e, fade, 0.0, 1.0), seed=1, stream_id=0, si=0)
                fb = rs.tx(sub_name, x, scen=(e, fade, 0.0, 1.0), seed=1, stream_id=1, si=1)
                cells = []
                for mode, with_sub in (("none", False), ("temporal", False)):
                    s = stats(main_name, fa, len(x), mode=mode)
                    cells.append(f"{100 * s['crc_fail'] / s['frames']:5.1f}")
                one = " / ".join(cells)
                cells = []
                for mode in MODES:
                    s = stats(main_name, fa, len(x), sub_name, fb, mode)
                    cells.append(f"{100 * s['crc_fail'] / s['frames']:5.1f}")
                print(f"{kind + ' ' + str(e) + ' dB':14}{one:>26}{' / '.join(cells):>26}", flush=True)


if __name__ == "__main__":
    main()
