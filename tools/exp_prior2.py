#!/usr/bin/env python3
"""
最低域 (k=0) の非保護bitの誤りを、前後と帯域間の相関 (非因果の線形予測) の事前で直す。
exp_prior_bits.py の続き。事前は学習データで作った固定の線形予測器 (非因果 L=8 + k=1 + k=2)。
隣の値は受信側の推定 (E0 → 事後平均) を使うので、実際の動作と同じ (正解は使わない)。反復は --iters 回。

  .venv/bin/python tools/exp_prior2.py --scenario 5 --frames 1200 --scale 1
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from exp_predict_gain import features, load  # noqa: E402
from exp_prior_bits import posterior_mean  # noqa: E402
from gracefm_ultra.rx import FloatRx  # noqa: E402
from gracefm.codec import offset_of, qmax_of, qmin_of  # noqa: E402

L = 8


def fit(tr):
    Xt, yt = [], []
    for s0, s1, s2 in tr:
        X, y = features(s0, s1, s2, L, True, True, True)
        Xt.append(X); yt.append(y)
    Xt, yt = np.concatenate(Xt), np.concatenate(yt)
    A = Xt.T @ Xt
    w = np.linalg.solve(A + 1e-6 * np.trace(A) / len(A) * np.eye(len(A)), Xt.T @ yt)
    res = yt - Xt @ w
    return w, float(np.mean(res ** 2) / np.mean(yt ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=sorted(glob.glob(os.path.join(HERE, "..", "data", "B200M", "val", "HOME*.npz")))[0])
    ap.add_argument("--scenario", type=int, default=5)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--scale", type=float, default=1.0, help="事前の分散の倍率")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--nu", type=float, default=0, help="Student-t の自由度 (0 ならガウス)")
    a = ap.parse_args()
    w, ratio = fit(load("train", 12, 6000))
    print(f"予測器: 特徴 {len(w)}、残差/信号 = {10 * np.log10(ratio):.1f} dB (学習データ平均)")
    d = np.load(a.file)
    si, F = a.scenario, a.frames
    print(f"scenario {si}: {d['scen'][si]}")
    rx = FloatRx(str(d["profile"]))
    lay, p = rx.lay, rx.p
    n0 = p.ncoef(0)
    items = [(i, t) for i, t in enumerate(lay.coef_items) if t[0] == "M" and t[1] == 0]
    b0 = p.bits_m[0]
    off, qmin, qmax = offset_of(b0, p, "M", 0), qmin_of(b0, p, "M", 0), qmax_of(b0)
    t0 = d["coef0"][:F, 0, :n0].astype(float)
    tq = d["coef"][:F, 0, :n0].astype(float)
    mean = np.zeros((F, 64))
    ok = np.zeros(F, bool)
    steps = np.ones((F, n0))
    p1s = [None] * F
    for f in range(F):
        if d["lost"][si, f]:
            continue
        r = rx.frame(d["llr"][si, f])
        if r["ok"]:
            ok[f] = True
            mean[f] = r["mean"][0]
            steps[f] = r["steps"][0, :n0]
            p1s[f] = r["p1"]
    E0 = mean[:, :n0].copy()
    cur = mean.copy()
    sig = (t0[ok] ** 2).sum()
    snr = lambda e: 10 * np.log10(sig / ((e[ok] - t0[ok]) ** 2).sum())  # noqa: E731
    print(f"CRC通過 {ok.mean():.3f}。最低域 (0-1 kHz) の SNR [dB] (CRCを通ったフレームのみ)")
    print(f"  量子化のみ(理想)      {snr(tq):7.2f}\n  E0 現行               {snr(E0):7.2f}")
    for it in range(a.iters):
        s0, s1, s2 = cur[:, 0:4].reshape(-1), cur[:, 4:8].reshape(-1), cur[:, 8:16].reshape(-1, 2)
        X, _ = features(s0, s1, s2, L, True, True, True)
        pred_full = np.zeros(len(s0))
        pred_full[L:len(s0) - L] = X @ w
        loc = uniform_filter1d(s0 ** 2, 33, mode="nearest")            # 局所の信号電力
        newm = cur[:, :n0].copy()
        for f in range(F):
            if not ok[f]:
                continue
            for j in range(n0):
                n = 4 * f + j
                var = max(ratio * loc[n] * a.scale, steps[f, j] ** 2 / 12)
                idx = [i for i, t in items if t[2] == j]
                pos = np.array([lay.coef_items[i][3] for i in idx])
                newm[f, j], _ = posterior_mean(p1s[f][idx], pos, off, steps[f, j], pred_full[n], var, qmin, qmax, nu=a.nu or None)
        cur[:, :n0] = newm
        print(f"  事後平均 (反復 {it + 1})     {snr(newm):7.2f}")


if __name__ == "__main__":
    main()
