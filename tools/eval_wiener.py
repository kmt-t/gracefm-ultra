#!/usr/bin/env python3
"""
学習を使わないウィナー推定の効果量を測る (係数領域)。Haar は直交なので、係数領域の誤差エネルギーは時間領域と同じ。
make_posteriors.py のキャッシュを読み、M 成分の SNR (正解は量子化前の係数 coef0) を Es/N0 の帯域ごとに出す。

  E0      現行と同じ (事後平均をそのまま使う。使えない係数は 0)
  E1(W)   ウィナー: 利得 σx²/(σx²+σn²)。σn² は事後分散、σx² は同じ帯域の ±W フレームの (平均² - 分散) の平均
  Oracle  σx² を正解係数から求める (E1 の上限の目安)
"""
import argparse
import glob
import os

import numpy as np

COUNTS = [4, 4, 8, 16, 32]
BAND = np.repeat(np.arange(5), COUNTS)


def local_energy(e, w, valid_w):
    """e: (F, 5) 帯域ごとの合計、valid_w: (F, 5) 係数数。±w フレームの窓で平均した (F, 5)"""
    k = np.ones(2 * w + 1)
    num = np.stack([np.convolve(e[:, b], k, "same") for b in range(5)], 1)
    den = np.stack([np.convolve(valid_w[:, b], k, "same") for b in range(5)], 1)
    return num / np.maximum(den, 1)


def band_sum(x):
    return np.stack([x[:, BAND == b].sum(1) for b in range(5)], 1)


def estimate(mean, var, truth, w, oracle=False):
    valid = np.isfinite(var)
    m = np.where(valid, mean, 0.0)
    v = np.where(valid, var, 0.0)
    src = truth ** 2 if oracle else (m ** 2 - v)
    e = band_sum(np.where(valid, src, 0.0))
    cnt = band_sum(valid.astype(float))
    sx = np.maximum(local_energy(e, w, cnt), 1e-3)[:, BAND]
    g = np.where(valid, sx / (sx + v), 0.0)
    return g * m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "B200M", "post", "val"))
    a = ap.parse_args()
    names = ["E0", "E1(2)", "E1(8)", "E1(25)", "Oracle(8)"]
    rows = []
    for f in sorted(glob.glob(os.path.join(a.data, "*.npz"))):
        d = np.load(f)
        S = d["mean"].shape[0]
        for s in range(S):
            mean, var = d["mean"][s][:, 0].astype(float), d["var"][s][:, 0].astype(float)
            t = d["coef0"][:len(mean), 0].astype(float)
            ok = d["ok"][s]
            ests = [np.where(np.isfinite(var), mean, 0.0), estimate(mean, var, t, 2), estimate(mean, var, t, 8),
                    estimate(mean, var, t, 25), estimate(mean, var, t, 8, oracle=True)]
            sig_all, sig_ok = (t ** 2).sum(), (t[ok] ** 2).sum()
            err_all = [((e - t) ** 2).sum() for e in ests]
            err_ok = [((e[ok] - t[ok]) ** 2).sum() for e in ests]
            rows.append([d["scen"][s][0], d["scen"][s][1], d["scen"][s][2], ok.mean(), sig_all, sig_ok] + err_all + err_ok)
    r = np.array(rows)
    print(f"{len(r)} シナリオ (E0=現行相当、M成分、係数領域のSNR [dB])\n")
    print("          Es/N0基準  n  CRC通過 | 全フレーム(欠落/CRC失敗は0)           | CRC通過フレームのみ")
    print("                                | " + " ".join(f"{n:>9}" for n in names) + " | " + " ".join(f"{n:>9}" for n in names))
    for fade in (0, 1):
        for lo in range(0, 18, 4):
            m = (r[:, 1] == fade) & (r[:, 0] >= lo) & (r[:, 0] < lo + 4)
            if not m.any():
                continue
            sa, so = r[m, 4].sum(), r[m, 5].sum()
            ea, eo = r[m, 6:11].sum(0), r[m, 11:16].sum(0)
            snr = lambda s, e: 10 * np.log10(s / np.maximum(e, 1e-30))  # noqa: E731
            print(f"{'rayleigh' if fade else 'awgn    '} {lo:2d}-{lo + 4:2d} dB {m.sum():2d}  {r[m, 3].mean():5.2f}   | "
                  + " ".join(f"{v:9.2f}" for v in snr(sa, ea)) + " | " + " ".join(f"{v:9.2f}" for v in snr(so, eo)))


if __name__ == "__main__":
    main()
