#!/usr/bin/env python3
"""
量子化だけ (通信路なし) の B200 出力で、誤差 (出力 - 原音) のスペクトルが、高域で原音に対してどれだけあるかを測る。
低域の量子化ノイズが Haar 合成で高域に漏れて、高域に信号がない箇所ではマスクされずに聞こえる、という問題の大きさの確認。
  帯域ごとに、原音のレベル、誤差のレベル (どちらも全体のフルスケールに対する dB)、その差 (正なら誤差が原音を上回る)。

  .venv/bin/python tools/exp_quant_leak.py --seconds 8
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from materials import MATERIALS, load  # noqa: E402
from gracefm_ultra.gfm import GRACEFM  # noqa: E402,F401
from gracefm.decoder import roundtrip  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402

BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()
    os.sched_setaffinity(0, {7, 11})
    p = PROFILES[a.profile]
    print(f"{a.profile}: 量子化のみ (通信路なし)。帯域ごとの 原音レベル / 誤差レベル [dBFS、帯域内の合計電力] / 誤差−原音 [dB] (正なら誤差の方が大きい = 漏れが聞こえうる)\n")
    print(f"{'素材':12}" + "".join(f"{lo // 1000}-{hi // 1000}kHz".rjust(22) for lo, hi in BANDS))
    for mat in MATERIALS:
        x = load(mat, a.seconds)
        xi = wavio.to_pcm24(x)
        y = roundtrip(xi, p)
        ref = xi.astype(float) / (1 << 23)
        err = (y.astype(float) - xi.astype(float)) / (1 << 23)
        cells = []
        for lo, hi in BANDS:
            P_r = P_e = 0.0
            for c in (0, 1):
                f, pr = welch(ref[:, c], 32000, nperseg=4096)
                _, pe = welch(err[:, c], 32000, nperseg=4096)
                m = (f >= lo) & (f < hi)
                P_r += pr[m].sum() * (f[1] - f[0]) / 2
                P_e += pe[m].sum() * (f[1] - f[0]) / 2
            r_db, e_db = 10 * np.log10(max(P_r, 1e-30)), 10 * np.log10(max(P_e, 1e-30))
            cells.append(f"{r_db:6.1f}/{e_db:6.1f}/{e_db - r_db:+6.1f}")
        print(f"{mat:12}" + "".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
