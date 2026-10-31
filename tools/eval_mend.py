#!/usr/bin/env python3
"""
MendNet の評価。検証データ (gen_hn.py --mend) の、使えないフレームの SNR [dB] を、状態 (壊れた・欠落) ごとに、
ゼロ埋め・AR 両側補間・MendNet で比べる。聴感は別に聴き比べる。

  .venv-gpu/bin/python tools/eval_mend.py --profile B200 --ckpt runs/mend_B200.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from gracefm_ultra.mendnet import WIN, MendNet, build_input, compose, haar_fwd, haar_inv  # noqa: E402
from gracefm_ultra.plc_ar import interpolate  # noqa: E402
from train_mend import Windows  # noqa: E402


def ar_fill(coef, usable):
    """窓の係数 (T,64) の、使えないフレームを、時間領域の AR 両側補間で埋める"""
    x = haar_inv(coef).reshape(-1)
    t = len(usable)
    i = 0
    while i < t:
        if usable[i]:
            i += 1
            continue
        j = i
        while j < t and not usable[j]:
            j += 1
        x[i * 64:j * 64] = interpolate(x, i * 64, j * 64)
        i = j
    return haar_fwd(x.reshape(-1, 64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", type=int, default=400)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    va = Windows(sorted(glob.glob(os.path.join(a.data, a.profile, "mend", "val", "*.npz"))), dev)
    ck = torch.load(a.ckpt, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    rng = np.random.default_rng(7)
    acc = {}                                                               # (状態, 手法) → [信号, 誤差]
    for _ in range(a.windows):
        b = va.batch(1, rng)
        with torch.no_grad():
            x, sig, usable = build_input(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"])
            est = compose(net(x), b["coef"], sig, usable)[0].cpu().numpy()
        t, c, st = b["target"][0].cpu().numpy(), b["coef"][0].cpu().numpy(), b["status"][0].cpu().numpy()
        us = st <= 1
        fills = {"ゼロ": c, "AR": ar_fill(c, us), "MendNet": est}
        for name, name_st in (("壊れた", 2), ("欠落", 3)):
            m = st == name_st
            if not m.any():
                continue
            for k, y in fills.items():
                s = acc.setdefault((name, k), [0.0, 0.0])
                s[0] += float((t[m] ** 2).sum())
                s[1] += float(((y[m] - t[m]) ** 2).sum())
    for name in ("壊れた", "欠落"):
        print(name, "  ".join(f"{k} {10 * np.log10(acc[(name, k)][0] / acc[(name, k)][1]):+6.2f} dB" for k in ("ゼロ", "AR", "MendNet") if (name, k) in acc))


if __name__ == "__main__":
    main()
