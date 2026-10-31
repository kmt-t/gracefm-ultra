#!/usr/bin/env python3
"""
欠落フレームを埋めた出力に、因果の IIR (Butterworth) を「欠落区間だけ」切り替えて通す。カットオフと次数の掃引。
IIR は信号全体に常時かけておき (状態は常に最新)、欠落区間 (前後 約 1 ms のクロスフェード) だけ出力を IIR 側にする。
群遅延 (約 2 サンプル) は整数遅れで補償する (実機では素通しの経路にも同じ遅れを入れる)。
埋め方: gfm = 現行 GraceFM の LPC 外挿、net = 欠落補間 DNN (両端クロスフェード付き)。欠落は 300 ms ごと。M 成分、32 kHz。

  .venv/bin/python tools/exp_iir_cutoff.py --seconds 8
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, group_delay, sos2tf, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_plc import BLOCK, FS, _gfm  # noqa: E402
from eval_plc_visqol import apply  # noqa: E402
from exp_wiener_post import MATERIALS, load  # noqa: E402
from tools.common import wavio  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}
ORDERS = [2, 4]
CUTS = [4000, 6000, 8000, 12000]


def iir_switch(y, mask, order, fc, xf=33):
    sos = butter(order, fc / (FS / 2), output="sos")
    b, a = sos2tf(sos)
    _, gd = group_delay((b, a), w=np.array([2 * np.pi * 1000 / FS]))
    D = int(round(float(gd[0])))
    yi = np.roll(sosfilt(sos, y), -D)                                  # 常時計算。群遅延ぶんを補償
    g = np.convolve(mask.astype(float), np.ones(xf) / xf, "same")
    return g * yi + (1 - g) * y


def job(args):
    mat, seconds, plan, orders, cuts = args
    os.sched_setaffinity(0, CPUS)
    from visqol import VisqolApi
    from gracefm_ultra.plcnet import load_predictor
    api = VisqolApi()
    api.create(mode="audio")
    x = load(mat, seconds).mean(1)
    ref = np.asarray(wavio.to_pcm24(x[:, None])[:, 0], float)
    ref48 = wavio.resample(ref / (1 << 23), FS, 48000)
    fills = {"gfm": _gfm, "net": load_predictor(os.path.join(HERE, "..", "runs", "plc2.pt"))}
    period = int(0.3 * FS) // BLOCK * BLOCK
    out = []
    for fill, g in plan:
        y = apply(ref, g, period, fills[fill])
        mask = np.zeros(len(ref), bool)
        for p0 in range(2048, len(ref) - 2048 - g * BLOCK, period):
            mask[p0:p0 + g * BLOCK] = True
        variants = {"none": y}
        for o in orders:
            for fc in cuts:
                variants[f"B{o}_{fc / 1000:g}k"] = iir_switch(y, mask, o, fc)
        for k, z in variants.items():
            snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - z) ** 2), 1e-9))
            mos = float(api.measure_from_arrays(ref48, wavio.resample(z / (1 << 23), FS, 48000), sample_rate=48000).moslqo)
            out.append((mat, fill, g, k, snr, mos))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--orders", default="2,4")
    ap.add_argument("--cuts", default="4000,6000,8000,12000")
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    ORDERS[:] = [int(v) for v in a.orders.split(",")]
    CUTS[:] = [int(v) for v in a.cuts.split(",")]
    plan = [("gfm", 3), ("gfm", 8), ("net", 8), ("net", 25)]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(job, [(m, a.seconds, plan, list(ORDERS), list(CUTS)) for m in MATERIALS]) for r in part]
    keys = ["none"] + [f"B{o}_{fc / 1000:g}k" for o in ORDERS for fc in CUTS]
    print(f"{len(MATERIALS)} 素材 × {a.seconds}s。ViSQOL / SNR[dB] の平均 (欠落区間だけ IIR に切り替え)")
    print(f"{'埋め方':>5}{'欠落':>6}" + "".join(f"{k:>13}" for k in keys))
    for fill, g in plan:
        sel = [r for r in rows if r[1] == fill and r[2] == g]
        print(f"{fill:>5}{g:>4}fr" + "".join(f"{np.mean([r[5] for r in sel if r[3] == k]):7.3f}/{np.mean([r[4] for r in sel if r[3] == k]):5.1f}" for k in keys))
    print("\n素材別 ViSQOL (net、8 フレーム欠落)")
    print(f"{'':12}" + "".join(f"{k:>9}" for k in keys))
    for m in MATERIALS:
        print(f"{m:12}" + "".join(f"{[r[5] for r in rows if r[0] == m and r[1] == 'net' and r[2] == 8 and r[3] == k][0]:9.3f}" for k in keys))
    print("\n素材別 ViSQOL (gfm、8 フレーム欠落)")
    for m in MATERIALS:
        print(f"{m:12}" + "".join(f"{[r[5] for r in rows if r[0] == m and r[1] == 'gfm' and r[2] == 8 and r[3] == k][0]:9.3f}" for k in keys))


if __name__ == "__main__":
    main()
