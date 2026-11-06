#!/usr/bin/env python3
"""
超解像の判定の閾値の確認。AWGN の Es/N0 ごとに、MendNet の出力の予測の SNR、判定がオンの割合、実際の SNR を出す。
予測と実際が近く、判定が使える C/N0 でオンになることを確かめる。

  .venv/bin/python tools/check_gate.py --profile B200-L
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm")))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.postnet import gate  # noqa: E402
from gracefm_ultra.transform import coef_power  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200-L")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    power = coef_power(PROFILES[a.profile].transform)
    w = np.tile(power, 2)
    acc = {}
    for f in sorted(glob.glob(os.path.join(a.data, a.profile, "post", "awgn_val", "*.npz"))):
        z = np.load(f)
        truth = z["truth"].astype(np.float64)
        for sc, es in enumerate(z["scen"][:, 0]):
            e0, sd, status = z["e0"][sc], z["sd"][sc], z["status"][sc]
            u = status <= 1
            s = acc.setdefault(float(es), [0.0, 0.0, 0.0, 0.0, []])
            s[0] += ((e0.astype(np.float64) ** 2 * w).sum(1) * u).sum()
            s[1] += ((sd.astype(np.float64) ** 2 * w).sum(1) * u).sum()
            s[2] += (truth ** 2 * w).sum()
            s[3] += ((e0 - truth) ** 2 * w).sum()
            s[4].append(gate(e0, sd, status, power).mean())
    print("Es/N0 [dB]  予測の SNR  判定オン  実際の SNR")
    for es, s in sorted(acc.items()):
        print(f"{es:9g}  {10 * np.log10(s[0] / s[1]):10.2f}  {np.mean(s[4]):8.0%}  {10 * np.log10(s[2] / s[3]):10.2f}")


if __name__ == "__main__":
    main()
