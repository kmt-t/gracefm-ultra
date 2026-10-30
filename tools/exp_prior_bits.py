#!/usr/bin/env python3
"""
最低域 (k=0) の非保護bitの誤りを、LPC の事前で直せるか (BitNet の解析版・効果量の見積もり)。
各係数について、候補の仮数 u (全 2^b 通り) に対し
    事後(u) ∝ Π_bit p(受信 | bit) × N(u·刻み; 予測, σp²)
を計算し、事後平均を出す。予測は過去の最低域係数列 (4係数/フレーム) の線形予測。

  prior=truth : 過去の値に正解の量子化値を使う (予測の上限。誤りが連鎖しない理想)
  prior=est   : 過去の値に自分の推定値を使う (現実の因果的な動作)
  oracle2     : 正解の前後 (±1フレーム) からの線形補間を事前にする (非因果で、200 ms バッファの使い方の例)
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.rx import FloatRx  # noqa: E402


def lpc_predict(hist, order, ridge=1e-9):
    """hist: 過去の系列 (最新が末尾)。自己相関法で次の値を予測し、(予測, 予測誤差の分散) を返す"""
    h = np.asarray(hist, float)
    n = len(h)
    if n <= order + 1 or not np.any(h):
        return 0.0, max(np.mean(h ** 2), 1.0) if n else 1.0
    r = np.array([np.dot(h[:n - k], h[k:]) for k in range(order + 1)])
    r[0] *= 1 + 1e-6
    r[0] += ridge
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    try:
        a = np.linalg.solve(R, r[1:order + 1])
    except np.linalg.LinAlgError:
        return float(h[-1]), float(np.mean(h ** 2))
    pred = float(np.dot(a, h[::-1][:order]))
    var = max(r[0] - np.dot(a, r[1:order + 1]), 1e-6) / n
    return pred, var


def posterior_mean(p1_bits, pos, offset, step, pred, var, qmin, qmax, nu=None):
    """p1_bits: 仮数bitのp1 (pos の並び)。候補 u の事後平均 (係数の単位) と事後分散を返す"""
    nb = int(pos.max()) + 1
    u = np.arange(1 << nb)
    ll = np.zeros(len(u))
    for pb, ps in zip(p1_bits, pos):
        bit = (u >> ps) & 1
        pb = np.clip(pb, 1e-6, 1 - 1e-6)
        ll += np.where(bit == 1, np.log(pb), np.log(1 - pb))
    m = u - offset
    x = m * step
    if nu is None:
        ll += -0.5 * (x - pred) ** 2 / var
    else:                                    # 裾の重い事前 (Student-t、自由度 nu)
        ll += -0.5 * (nu + 1) * np.log1p((x - pred) ** 2 / (nu * var))
    ll[(m < qmin) | (m > qmax)] = -np.inf
    w = np.exp(ll - ll.max())
    w /= w.sum()
    mu = (w * x).sum()
    return mu, (w * (x - mu) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=sorted(glob.glob(os.path.join(HERE, "..", "data", "B200M", "val", "*.npz")))[0])
    ap.add_argument("--scenario", type=int, default=-1, help="-1 なら Es/N0 が 10 dB 以上の AWGN で欠落なしのものを選ぶ")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--hist", type=int, default=64)
    ap.add_argument("--var-scale", type=float, default=1.0, help="事前の分散の倍率 (大きいと事前が弱まり E0 に近づく)")
    a = ap.parse_args()
    d = np.load(a.file)
    si = a.scenario
    if si < 0:
        si = next(s for s in range(d["scen"].shape[0]) if d["scen"][s][0] >= 10 and d["scen"][s][1] == 0 and d["scen"][s][2] == 0)
    print(f"scenario {si}: {d['scen'][si]}")
    rx = FloatRx(str(d["profile"]))
    lay, p = rx.lay, rx.p
    F = a.frames
    n0 = p.ncoef(0)
    # 最低域 (M, k=0) の仮数bit: coef_items = (ch, k, j, pos)
    items = [(i, t) for i, t in enumerate(lay.coef_items) if t[0] == "M" and t[1] == 0]
    from gracefm.codec import offset_of, qmax_of, qmin_of
    b0 = p.bits_m[0]
    off, qmin, qmax = offset_of(b0, p, "M", 0), qmin_of(b0, p, "M", 0), qmax_of(b0)
    t0 = d["coef0"][:F, 0, :n0].astype(float)        # 量子化前
    tq = d["coef"][:F, 0, :n0].astype(float)         # 量子化後の正解
    est0 = np.zeros((F, n0))
    ok = np.zeros(F, bool)
    steps = np.ones((F, n0))
    p1s = [None] * F
    for f in range(F):
        if d["lost"][si, f]:
            continue
        r = rx.frame(d["llr"][si, f])
        if r["ok"]:
            ok[f] = True
            est0[f] = r["mean"][0, :n0]
            steps[f] = r["steps"][0, :n0]
            p1s[f] = r["p1"]
    seqs = {"truth": tq.copy(), "est": est0.copy()}
    res = {"E0": est0.copy(), "truth": np.zeros((F, n0)), "est": np.zeros((F, n0)), "oracle2": np.zeros((F, n0))}
    for mode in ("truth", "est", "oracle2"):
        hist = []
        for f in range(F):
            for j in range(n0):
                if not ok[f]:
                    hist.append(res["est" if mode != "truth" else "truth"][f, j] if False else 0.0)
                    continue
                if mode == "oracle2":
                    lo = tq[f - 1, j] if f > 0 else 0.0
                    hi = tq[f + 1, j] if f + 1 < F else 0.0
                    pred, var = 0.5 * (lo + hi), (0.25 * (lo - hi) ** 2 + (steps[f, j]) ** 2) + 1.0
                else:
                    h = np.array(hist[-a.hist * 1:])
                    pred, var = lpc_predict(h, a.order)
                    var = max(var, steps[f, j] ** 2 / 12) * a.var_scale
                idx = [i for i, t in items if t[2] == j]
                pos = np.array([lay.coef_items[i][3] for i in idx])
                mu, _ = posterior_mean(p1s[f][idx], pos, off, steps[f, j], pred, var, qmin, qmax)
                res[mode][f, j] = mu
                hist.append(tq[f, j] if mode == "truth" else mu)
    sig = (t0[ok] ** 2).sum()
    print(f"CRC通過 {ok.mean():.3f}、最低域 (0-1 kHz) の SNR [dB] (CRCを通ったフレームのみ):")
    for name, e in (("量子化のみ(理想)", tq), ("E0 現行", res["E0"]), ("LPC事前(過去=正解)", res["truth"]),
                    ("LPC事前(過去=自推定)", res["est"]), ("前後補間(正解)事前", res["oracle2"])):
        print(f"  {name:22} {10 * np.log10(sig / ((e[ok] - t0[ok]) ** 2).sum()):7.2f}")


if __name__ == "__main__":
    main()
