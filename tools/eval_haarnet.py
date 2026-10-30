#!/usr/bin/env python3
"""
HaarNet-lite の評価 (検証データ、M 成分 k=0 の SNR)。生LLRの事後平均 E0 と比べる。欠落フレームは除く。
通信路 (AWGN/Rayleigh) と Es/N0 の帯域ごとに、誤差エネルギーを合算して SNR を出す。
比較用に、FEC ありの E0 と Student-t 事前 (eval_prior_multi.py) の最低域の値も、同じ表の形で載せる。

  .venv/bin/python tools/eval_haarnet.py --ckpt runs/hn.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.haarnet import WIN, HaarNetLite, features, make_meta  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "runs", "hn.pt"))
    ap.add_argument("--frames", type=int, default=10000)
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(len(CPUS))
    ck = torch.load(a.ckpt, map_location="cpu")
    net = HaarNetLite(**ck["cfg"]).eval()
    net.load_state_dict(ck["model"])
    rows = []
    edge = 8
    for f in sorted(glob.glob(os.path.join(HERE, "..", "data", "B200M", "hn", "val", "*.npz"))):
        d = np.load(f)
        meta = make_meta(d, PROFILES["B200M"])
        F = min(a.frames, d["llr"].shape[1])
        for s in range(d["llr"].shape[0]):
            sig = err0 = err1 = 0.0
            starts = list(range(0, F - WIN + 1, WIN - 2 * edge))
            for b0 in range(0, len(starts), 32):
                ss = starts[b0:b0 + 32]
                cut = lambda arr: torch.from_numpy(np.stack([arr[t:t + WIN] for t in ss]))  # noqa: E731
                llr, var = cut(d["llr"][s].astype(np.float32)), cut(d["var"][s].astype(np.float32))
                lost, step, truth = cut(d["lost"][s]), cut(d["step"]), cut(d["coef0"])
                with torch.no_grad():
                    x, S, e0n = features(llr, step, var, lost, meta)
                    est = (e0n + net(x)) * S
                    e0 = e0n * S
                keep = (~lost).float()[..., None].clone()
                keep[:, :edge] = 0
                keep[:, -edge:] = 0
                k0 = keep[..., :1]
                sig += float(((truth[..., :4] ** 2) * k0).sum())
                err0 += float((((e0 - truth)[..., :4] ** 2) * k0).sum())
                err1 += float((((est - truth)[..., :4] ** 2) * k0).sum())
            rows.append((d["scen"][s][0], d["scen"][s][1], sig, err0, err1))
    r = np.array(rows)
    print("最低域 (0-1 kHz) の SNR [dB]。欠落フレームを除く全フレーム (CRC 失敗フレームも含む)")
    print("                         n      E0(生LLR)   HaarNet-lite   改善")
    for fade in (0, 1):
        for lo, hi in ((0, 6), (6, 10), (10, 14), (14, 20)):
            m = (r[:, 1] == fade) & (r[:, 0] >= lo) & (r[:, 0] < hi)
            if not m.any():
                continue
            s_, e0, e1 = r[m, 2].sum(), r[m, 3].sum(), r[m, 4].sum()
            print(f"{'rayleigh' if fade else 'awgn    '} {lo:2d}-{hi:2d} dB {m.sum():3d}   {10 * np.log10(s_ / e0):10.2f} {10 * np.log10(s_ / e1):14.2f} {10 * np.log10(e0 / e1):+8.2f}")


if __name__ == "__main__":
    main()
