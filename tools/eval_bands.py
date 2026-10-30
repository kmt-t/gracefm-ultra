#!/usr/bin/env python3
"""
領域A' (C/N が良い通信路、CRC を通ったフレームだけ): 帯域別の SNR を、現行・ウィナー・量子化だけの理想と比べる。
量子化だけの理想 = 正解の量子化係数 (coef) と量子化前の係数 (coef0) の差。

  python tools/eval_bands.py [--min-esn0 10] [--min-ok 0.95]
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_wiener import BAND, estimate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "B200M", "post", "val"))
    ap.add_argument("--min-esn0", type=float, default=10.0)
    ap.add_argument("--min-ok", type=float, default=0.95)
    ap.add_argument("--fade", type=int, default=0, help="0: awgn, 1: rayleigh")
    a = ap.parse_args()
    names = ["量子化のみ", "E0(現行)", "E1(8)", "Oracle(8)"]
    sig = np.zeros(5)
    err = np.zeros((len(names), 5))
    used = 0
    for f in sorted(glob.glob(os.path.join(a.data, "*.npz"))):
        d = np.load(f)
        for s in range(d["mean"].shape[0]):
            e0, fade, rate, _ = d["scen"][s]
            ok = d["ok"][s] & d["high_ok"][s]
            if fade != a.fade or e0 < a.min_esn0 or ok.mean() < a.min_ok:
                continue
            used += 1
            mean, var = d["mean"][s][:, 0].astype(float), d["var"][s][:, 0].astype(float)
            t = d["coef0"][:len(mean), 0].astype(float)
            q = d["coef"][:len(mean), 0].astype(float)
            ests = [q, np.where(np.isfinite(var), mean, 0.0), estimate(mean, var, t, 8), estimate(mean, var, t, 8, oracle=True)]
            for b in range(5):
                m = BAND == b
                sig[b] += (t[ok][:, m] ** 2).sum()
                for i, e in enumerate(ests):
                    err[i, b] += ((e[ok][:, m] - t[ok][:, m]) ** 2).sum()
    print(f"{used} シナリオ ({'rayleigh' if a.fade else 'awgn'}、Es/N0基準 >= {a.min_esn0} dB、CRC通過率 >= {a.min_ok})、CRCを通ったフレームのみ\n")
    print(f"{'':12}" + "".join(f"{e:>9}" for e in ["0-1k", "1-2k", "2-4k", "4-8k", "8-16k"]) + f"{'全体':>9}")
    for i, n in enumerate(names):
        snr = 10 * np.log10(sig / np.maximum(err[i], 1e-30))
        print(f"{n:12}" + "".join(f"{v:9.2f}" for v in snr) + f"{10 * np.log10(sig.sum() / err[i].sum()):9.2f}")
    print("\n帯域のエネルギー比 [dB]: " + " ".join(f"{10 * np.log10(s / sig.sum()):6.1f}" for s in sig))
    print("誤差エネルギーの帯域内訳 [%] (E0): " + " ".join(f"{100 * e / err[1].sum():5.1f}" for e in err[1]))


if __name__ == "__main__":
    main()
