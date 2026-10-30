#!/usr/bin/env python3
"""
最低域 (k=0) の係数を、文脈からどこまで予測できるか (事前として使える精度の上限の見積もり)。
学習データの正解の量子化係数で線形予測器 (最小二乗) を作り、検証データで予測利得 [dB] を測る。
予測利得 = 信号電力 / 予測誤差電力。誤差は -26 dB 付近にあるので、その近くまで届けば bit 判定の助けになる。

  文脈: 同じ帯域の前 (因果) / 前後 (非因果) の L 点、他の帯域 (k=1,2) の同じ時刻と前後
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "..", "data", "B200M")


def load(split, nfile, nframe):
    out = []
    for f in sorted(glob.glob(os.path.join(SPLIT, split, "*.npz")))[:nfile]:
        c = np.load(f)["coef"][:nframe, 0].astype(float)     # (F, 64) M成分の量子化係数
        out.append((c[:, 0:4].reshape(-1), c[:, 4:8].reshape(-1), c[:, 8:16].reshape(-1, 2)))
    return out


def features(s0, s1, s2, L, two_sided, use_b1, use_b2):
    n = len(s0)
    cols = [np.roll(s0, k) for k in range(1, L + 1)]                 # 過去
    if two_sided:
        cols += [np.roll(s0, -k) for k in range(1, L + 1)]           # 未来
    if use_b1:
        cols += [np.roll(s1, k) for k in range(-L // 2, L // 2 + 1)]
    if use_b2:
        cols += [np.roll(s2[:, 0], k) for k in range(-2, 3)]
    X = np.stack(cols, 1)
    return X[L:n - L], s0[L:n - L]


def run(name, L, two_sided, use_b1, use_b2, tr, va):
    Xt, yt = [], []
    for s0, s1, s2 in tr:
        X, y = features(s0, s1, s2, L, two_sided, use_b1, use_b2)
        Xt.append(X); yt.append(y)
    Xt, yt = np.concatenate(Xt), np.concatenate(yt)
    w = np.linalg.lstsq(Xt.T @ Xt + 1e-6 * np.trace(Xt.T @ Xt) / Xt.shape[1] * np.eye(Xt.shape[1]), Xt.T @ yt, rcond=None)[0]
    e = s = 0.0
    for s0, s1, s2 in va:
        X, y = features(s0, s1, s2, L, two_sided, use_b1, use_b2)
        e += ((y - X @ w) ** 2).sum(); s += (y ** 2).sum()
    print(f"  {name:38} L={L:2d} 特徴 {Xt.shape[1]:3d}  予測利得 {10 * np.log10(s / e):6.2f} dB")


def main():
    tr, va = load("train", 12, 6000), load("val", 4, 6000)
    print(f"学習 {len(tr)} 音源 / 検証 {len(va)} 音源 (各 6000 フレーム)。固定の線形予測器 (全体で1つ)\n")
    for L in (8, 16, 32):
        run("因果 (過去のみ)", L, False, False, False, tr, va)
    for L in (8, 16, 32):
        run("非因果 (前後)", L, True, False, False, tr, va)
    for L in (8, 16):
        run("非因果 + k=1 の同時刻前後", L, True, True, False, tr, va)
        run("非因果 + k=1 + k=2", L, True, True, True, tr, va)


if __name__ == "__main__":
    main()
