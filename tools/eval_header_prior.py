#!/usr/bin/env python3
"""
ヘッダの事前知識つき復号（engine/gfm-rx の --prior）の効果: 基本 CRC が通らないフレームの割合を、Es/N0 ごとに、
事前知識なし（none）と、フレーム番号と直前の指数（full）で比べる。

  .venv/bin/python tools/eval_header_prior.py [--seconds 6]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import rs  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402

MODES = ("none", "full")


def stats(profile, frames, n, mode):
    """gfm-rx の標準エラー出力から、統計を読む"""
    with tempfile.TemporaryDirectory() as d:
        a, o = os.path.join(d, "a.sym"), os.path.join(d, "o.raw")
        rs.write_sym(a, frames)
        r = subprocess.run([os.path.join(rs.BIN, "gfm-rx"), "--profile", profile, "--frames-a", a, "--n", str(n), "--out", o, "--prior", mode],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-300:])
    return {k: int(v) for k, v in re.findall(r"(frames|lost|crc_fail|high_crc_fail|recovered) (\d+)", r.stderr)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--profiles", default="B200,B200M,M700")
    ap.add_argument("--esn0", default="2,4,6,8,10")
    ap.add_argument("--kinds", default="awgn,rayleigh")
    a = ap.parse_args()
    for name in a.profiles.split(","):
        x = rs.clip(PROFILES[name], a.seconds)
        print(f"\n{name}: 基本 CRC 不一致の割合 [%]（{' / '.join(MODES)}）\n")
        for kind in a.kinds.split(","):
            for e in [float(v) for v in a.esn0.split(",")]:
                frames = rs.tx(name, x, scen=(e, int(kind == "rayleigh"), 0.0, 1.0), seed=1)
                cells = [f"{100 * stats(name, frames, len(x), m)['crc_fail'] / len(frames):5.1f}" for m in MODES]
                print(f"{kind + ' ' + str(e) + ' dB':16}{' / '.join(cells)}", flush=True)


if __name__ == "__main__":
    main()
