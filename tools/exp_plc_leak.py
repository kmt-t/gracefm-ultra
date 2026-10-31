#!/usr/bin/env python3
"""
欠落区間を埋めたときの、高域の漏れ (誤差 − 原音の帯域別レベル、欠落区間だけ)。埋め方ごとの比較。
  zero / gfm (現行 LPC 外挿) / ar2 (AR 両側補間) / net (DNN、クロスフェードなし) / net+IIR (DNN + 因果 Butterworth 4 次 4 kHz の切り替え)
値は 欠落区間の誤差 − 欠落区間の原音 [dB] (帯域: 2-4k / 4-8k / 8-16k)。正なら誤差の方が大きく、隠れない。

  .venv/bin/python tools/exp_plc_leak.py --gap 8
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import butter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_plc import BLOCK, FS, _gfm  # noqa: E402
from eval_plc_visqol import apply  # noqa: E402
from gracefm_ultra.iir_switch import iir_switch  # noqa: E402
from materials import MATERIALS, load  # noqa: E402
from gracefm_ultra.plc_ar import interpolate  # noqa: E402
from gracefm_ultra.plcnet import load_predictor  # noqa: E402
from tools.common import wavio  # noqa: E402

BANDS = ((2000, 4000), (4000, 8000), (8000, 16000))


def band_db(ref, y, g, period):
    """欠落区間ごとの周期図 (Hann) を足して、帯域ごとの 誤差 − 原音 [dB]"""
    n = g * BLOCK
    win = np.hanning(n)
    fr = np.fft.rfftfreq(n, 1 / FS)
    Pe = np.zeros(len(fr))
    Pr = np.zeros(len(fr))
    for p0 in range(2048, len(ref) - 2048 - n, period):
        Pe += np.abs(np.fft.rfft(win * (y[p0:p0 + n] - ref[p0:p0 + n]))) ** 2
        Pr += np.abs(np.fft.rfft(win * ref[p0:p0 + n])) ** 2
    out = []
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        out.append(10 * np.log10(max(Pe[m].sum(), 1e-30) / max(Pr[m].sum(), 1e-30)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()
    os.sched_setaffinity(0, {7, 11})
    net0 = load_predictor(os.path.join(HERE, "..", "runs", "baseline", "plc.pt"), xfade=0)
    fills = {"zero": lambda x, p, q: np.zeros(q - p), "gfm": _gfm, "ar2": lambda x, p, q: interpolate(x, p, q), "net": net0}
    period = int(0.3 * FS) // BLOCK * BLOCK
    names = list(fills) + ["net+IIR"]
    print(f"欠落 {a.gap} フレーム ({a.gap * BLOCK / FS * 1000:.0f} ms)、300 ms ごと。欠落区間の 誤差−原音 [dB] (2-4k / 4-8k / 8-16k)。正なら隠れない\n")
    print(f"{'素材':12}" + "".join(f"{k:>22}" for k in names))
    acc = {k: [] for k in names}
    for mat in MATERIALS:
        x = load(mat, a.seconds).mean(1)
        ref = np.asarray(wavio.to_pcm24(x[:, None])[:, 0], float)
        mask = np.zeros(len(ref), bool)
        for p0 in range(2048, len(ref) - 2048 - a.gap * BLOCK, period):
            mask[p0:p0 + a.gap * BLOCK] = True
        row = []
        for k in names:
            y = apply(ref, a.gap, period, fills["net" if k == "net+IIR" else k])
            if k == "net+IIR":
                y = iir_switch(y, mask, 4, 4000)
            v = band_db(ref, y, a.gap, period)
            acc[k].append(v)
            row.append(f"{v[0]:+6.1f}/{v[1]:+6.1f}/{v[2]:+6.1f}")
        print(f"{mat:12}" + "".join(f"{c:>22}" for c in row), flush=True)
    print(f"{'平均':12}" + "".join(f"{'/'.join(f'{np.mean([v[i] for v in acc[k]]):+6.1f}' for i in range(3)):>22}" for k in names))


if __name__ == "__main__":
    main()
