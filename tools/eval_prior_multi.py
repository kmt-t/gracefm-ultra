#!/usr/bin/env python3
"""
exp_prior2 の方式 (非因果+帯域間の線形予測を Student-t 事前にした bit 判定) を、検証データの全シナリオで評価する。
最低域 (k=0) の SNR を、現行 (E0) と比べる。CRC を通ったフレームだけ。通信路で分けて集計する。

  .venv/bin/python tools/eval_prior_multi.py --frames 1000 --workers 6
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from exp_predict_gain import load  # noqa: E402
from exp_prior2 import fit, run  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}
_W = {}


def init(scale, nu, iters):
    os.sched_setaffinity(0, CPUS)
    w, ratio = fit(load("train", 12, 6000))
    _W.update(w=w, ratio=ratio, scale=scale, nu=nu, iters=iters)


def job(args):
    path, si, F = args
    return os.path.basename(path), run(path, si, F, _W["w"], _W["ratio"], _W["scale"], _W["nu"], _W["iters"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=0.04)
    ap.add_argument("--nu", type=float, default=3)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    files = sorted(glob.glob(os.path.join(HERE, "..", "data", "B200M", "val", "*.npz")))
    jobs = [(f, s, a.frames) for f in files for s in range(np.load(f)["llr"].shape[0])]
    rows = []
    with Pool(a.workers, initializer=init, initargs=(a.scale, a.nu or None, a.iters)) as pool:
        for name, r in pool.imap_unordered(job, jobs):
            rows.append((name, r))
            print(f"{name[:24]:24} Es/N0 {r['scen'][0]:5.1f} fade {int(r['scen'][1])} CRC通過 {r['ok']:.2f}", flush=True)
    print("\n=== 集計 (最低域の SNR [dB]、CRC を通ったフレームのみ。通信路ごとに誤差エネルギーを合算) ===")
    print("                            n  CRC通過  量子化のみ   E0(現行)   事前(反復1)  事前(反復2)  改善(反復2)")
    for fade in (0, 1):
        for lo, hi in ((0, 6), (6, 10), (10, 14), (14, 20)):
            sel = [r for _, r in rows if "err_e0" in r and int(r["scen"][1]) == fade and lo <= r["scen"][0] < hi]
            if not sel:
                continue
            sig = sum(r["sig"] for r in sel)
            db = lambda e: 10 * np.log10(sig / e)  # noqa: E731
            q, e0 = sum(r["err_q"] for r in sel), sum(r["err_e0"] for r in sel)
            its = [sum(r["err_it"][i] for r in sel) for i in range(len(sel[0]["err_it"]))]
            print(f"{'rayleigh' if fade else 'awgn    '} {lo:2d}-{hi:2d} dB {len(sel):3d}  {np.mean([r['ok'] for r in sel]):6.2f}  "
                  f"{db(q):9.2f} {db(e0):10.2f} " + " ".join(f"{db(e):11.2f}" for e in its) + f" {db(its[-1]) - db(e0):+11.2f}")
    low = [r for _, r in rows if "err_e0" not in r]
    print(f"\nCRC 通過が 50 フレーム未満で集計から除外したシナリオ: {len(low)} / {len(rows)}")


if __name__ == "__main__":
    main()
