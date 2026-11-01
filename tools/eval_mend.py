#!/usr/bin/env python3
"""
MendNet の評価。検証データ (gen_hn.py --mend) の、使えないフレームの SNR [dB] を、使えない連続区間の長さごとに、
ゼロ埋め・AR 両側補間・MendNet で比べる。聴感は別に聴き比べる。

  .venv/bin/python tools/eval_mend.py --profile B200 --ckpt runs/mend_B200.pt
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
from gracefm_ultra.mendnet import WIN, MendNet, build_input, compose, haar_fwd, haar_inv, join_ar, unusable_runs  # noqa: E402
from gracefm_ultra.plc_ar import interpolate  # noqa: E402
from train_mend import Windows  # noqa: E402


BUCKETS = [("連続1", 1, 1), ("連続2-3", 2, 3), ("連続4-8", 4, 8), ("連続9-25", 9, 25), ("連続26以上", 26, 10**9)]


CPUS = {int(v) for v in os.environ.get("ULTRA_CPUS", "0,1,5,6,7,11").split(",")}


def ar_fill(coef, usable):
    """窓の係数 (T,64) の、使えないフレームを、時間領域の AR 両側補間で埋める"""
    x = haar_inv(coef).reshape(-1)
    for i, j in unusable_runs(usable):
        x[i * 64:j * 64] = interpolate(x, i * 64, j * 64)
    return haar_fwd(x.reshape(-1, 64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", type=int, default=400)
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    va = Windows(sorted(glob.glob(os.path.join(a.data, a.profile, "mend", "val", "*.npz"))), dev)
    ck = torch.load(a.ckpt, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    refine = ck.get("refine", False)
    rng = np.random.default_rng(7)
    keep = [0.0, 0.0, 0.0]                                                  # 使えるフレーム: 信号、復号のまま、補正後の誤差
    acc = {}                                                               # (状態, 手法) → [信号, 誤差]
    for _ in range(a.windows):
        b = va.batch(1, rng)
        with torch.no_grad():
            x, sig, usable = build_input(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], *((b["hint_h"], b["hint_q"]) if ck.get("hint", False) else ()))
            est = compose(net(x), b["coef"], sig, usable, refine)[0].cpu().numpy()
        t, c, st = b["target"][0].cpu().numpy(), b["coef"][0].cpu().numpy(), b["status"][0].cpu().numpy()
        us = st <= 1
        fills = {"ゼロ": c, "AR": ar_fill(c, us), "MendNet": est, "MendNet+AR接続": join_ar(est, us)}
        keep[0] += float((t[us] ** 2).sum())
        keep[1] += float(((c[us] - t[us]) ** 2).sum())
        keep[2] += float(((est[us] - t[us]) ** 2).sum())
        bad = ~us
        run = np.zeros(len(us), int)                                       # 各フレームが属する、使えない連続区間の長さ
        i = 0
        while i < len(us):
            if us[i]:
                i += 1
                continue
            j = i
            while j < len(us) and not us[j]:
                j += 1
            run[i:j] = j - i
            i = j
        for name, lo, hi in BUCKETS:
            m = bad & (run >= lo) & (run <= hi)
            if not m.any():
                continue
            for k, y in fills.items():
                s = acc.setdefault((name, k), [0.0, 0.0])
                s[0] += float((t[m] ** 2).sum())
                s[1] += float(((y[m] - t[m]) ** 2).sum())
    for name, _, _ in BUCKETS:
        print(f"{name:10s}", "  ".join(f"{k} {10 * np.log10(acc[(name, k)][0] / acc[(name, k)][1]):+6.2f} dB" for k in ("ゼロ", "AR", "MendNet", "MendNet+AR接続") if (name, k) in acc))

    print(f"使えるフレーム  復号のまま {10 * np.log10(keep[0] / keep[1]):+6.2f} dB  補正後 {10 * np.log10(keep[0] / keep[2]):+6.2f} dB")


if __name__ == "__main__":
    main()
