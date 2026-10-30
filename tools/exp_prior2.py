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


def run(path, si, F, w, ratio, scale, nu, iters, verbose=False):
    """1 シナリオを評価。返り値: dict(ok=CRC通過率, q=量子化のみ, e0=現行, it=[反復ごとの SNR]) (最低域、CRC通過フレーム)"""
    d = np.load(path)
    F = min(F, d["llr"].shape[1])
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
    out = {"ok": float(ok.mean()), "scen": d["scen"][si].tolist(), "n": int(ok.sum())}
    if ok.sum() < 50:
        return out
    E0 = mean[:, :n0].copy()
    cur = mean.copy()
    sig = (t0[ok] ** 2).sum()
    err = lambda e: float(((e[ok] - t0[ok]) ** 2).sum())  # noqa: E731
    out.update(sig=float(sig), err_q=err(tq), err_e0=err(E0), err_it=[])
    for it in range(iters):
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
                var = max(ratio * loc[n] * scale, steps[f, j] ** 2 / 12)
                idx = [i for i, t in items if t[2] == j]
                pos = np.array([lay.coef_items[i][3] for i in idx])
                newm[f, j], _ = posterior_mean(p1s[f][idx], pos, off, steps[f, j], pred_full[n], var, qmin, qmax, nu=nu)
        cur[:, :n0] = newm
        out["err_it"].append(err(newm))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=sorted(glob.glob(os.path.join(HERE, "..", "data", "B200M", "val", "HOME*.npz")))[0])
    ap.add_argument("--scenario", type=int, default=5)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--scale", type=float, default=0.04, help="事前の分散の倍率")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--nu", type=float, default=3, help="Student-t の自由度 (0 ならガウス)")
    a = ap.parse_args()
    w, ratio = fit(load("train", 12, 6000))
    r = run(a.file, a.scenario, a.frames, w, ratio, a.scale, a.nu or None, a.iters)
    db = lambda e: 10 * np.log10(r["sig"] / e)  # noqa: E731
    print(f"scenario {a.scenario}: {r['scen']}  CRC通過 {r['ok']:.3f}")
    print(f"  量子化のみ(理想) {db(r['err_q']):7.2f}\n  E0 現行          {db(r['err_e0']):7.2f}")
    for i, e in enumerate(r["err_it"]):
        print(f"  事後平均 (反復{i + 1}) {db(e):7.2f}")


if __name__ == "__main__":
    main()
