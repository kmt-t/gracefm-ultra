#!/usr/bin/env python3
"""
data/<profile>/<split>/*.npz の受信LLRを FloatRx で復号し、係数の事後平均・分散を data/<profile>/post/<split>/ に保存する。
推定器の実験を FEC 復号なしで高速に繰り返すためのキャッシュ。

  .venv/bin/python tools/make_posteriors.py --split val --frames 5000 --workers 6
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.rx import FloatRx  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}      # fireball 側の測定と共有しているので、使ってよいコアはここだけ


def work(job):
    """(ファイル, シナリオ) 1つぶんを復号する"""
    path, si, frames = job
    os.sched_setaffinity(0, CPUS)
    d = np.load(path)
    rx = FloatRx(str(d["profile"]))
    F = min(d["llr"].shape[1], frames) if frames else d["llr"].shape[1]
    llr, lost = d["llr"][si, :F], d["lost"][si, :F]
    mean = np.zeros((F, 2, rx.block), np.float32)
    var = np.full((F, 2, rx.block), np.inf, np.float32)
    ok = np.zeros(F, bool)
    high_ok = np.zeros(F, bool)
    for f in range(F):
        if lost[f]:
            continue
        r = rx.frame(llr[f])
        ok[f], high_ok[f] = r["ok"], r["high_ok"]
        if r["ok"]:
            mean[f], var[f] = r["mean"], r["var"]
    return path, si, mean, var, ok, high_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M")
    ap.add_argument("--split", default="val")
    ap.add_argument("--frames", type=int, default=0, help="先頭から何フレームか (0 なら全部)")
    ap.add_argument("--files", type=int, default=0, help="先頭から何ファイルか (0 なら全部)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    files = sorted(glob.glob(os.path.join(a.data, a.profile, a.split, "*.npz")))
    if a.files:
        files = files[:a.files]
    outdir = os.path.join(a.data, a.profile, "post", a.split)
    os.makedirs(outdir, exist_ok=True)
    jobs = []
    for f in files:
        n_scen = np.load(f)["llr"].shape[0]
        jobs += [(f, si, a.frames) for si in range(n_scen)]
    res = {}
    t0 = time.time()
    with Pool(min(a.workers, len(jobs), len(CPUS))) as pool:
        for path, si, mean, var, ok, high_ok in pool.imap_unordered(work, jobs):
            res.setdefault(path, {})[si] = (mean, var, ok, high_ok)
            print(f"{os.path.basename(path)} scenario {si}: done ({time.time() - t0:.0f}s)", flush=True)
    for path, r in res.items():
        d = np.load(path)
        S = len(r)
        F = r[0][0].shape[0]
        stack = lambda i: np.stack([r[si][i] for si in range(S)])  # noqa: E731
        np.savez(os.path.join(outdir, os.path.basename(path)), mean=stack(0), var=stack(1), ok=stack(2), high_ok=stack(3),
                 lost=d["lost"][:, :F], esn0=d["esn0"][:, :F], coef=d["coef"][:F], coef0=d["coef0"][:F],
                 scen=d["scen"], bits=d["bits"][:F])


if __name__ == "__main__":
    main()
