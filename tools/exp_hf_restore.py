#!/usr/bin/env python3
"""
高域 (k=3,4 = 4〜16 kHz) が欠けたとき、低域 (k=0..2) などからウィナー (線形最小二乗) 推定で復元できるかの評価。
通信路は通さず、正解の Haar 係数から高域を消して試す (ビット誤りとは切り離す)。

  S1  全フレームで高域を消す (副系・高域なしの受信の最悪ケース)。入力: 低域の係数 (前後 ±2 フレーム)
  S2  20% のフレームで孤立して高域を消す (高域 CRC 不一致)。入力: 低域 (前後 ±2) + 前後フレームの高域 (届いているもの)
推定器は学習用音源 (検証用・ambient を除く) の 係数から作る 線形回帰 (リッジ)。尺度は、窓内の低域の rms で正規化する。
比較: zero (現行)、zero+IIR (因果 Butterworth 4 次 6 kHz)、lmmse (ウィナー)、lmmse+IIR。
指標: 高域の SNR (係数領域、k=3,4 のみ)、全体の SNR、ViSQOL。

  .venv/bin/python tools/exp_hf_restore.py --seconds 8
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import butter, sosfilt, welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from exp_wiener_post import FS, GRACEFM, MATERIALS, load  # noqa: E402
from gracefm_ultra.plcnet import haar_fwd, haar_inv  # noqa: E402
from tools.common import wavio  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}
LF, HF = slice(0, 16), slice(16, 64)
CTX = 2


def coefs_of(m):
    n = len(m) // 64
    return haar_fwd(np.asarray(m[:n * 64], float).reshape(n, 64))


def feats(C, hf_avail=None, ctx=CTX):
    """C: (F, 64)。低域の前後 ±ctx フレーム (正規化)、hf_avail があれば前後 1 フレームの高域も。返り値: (X, 尺度 s (F,))"""
    F = len(C)
    lf = C[:, LF]
    pad = np.pad(lf, ((ctx, ctx), (0, 0)))
    win = np.concatenate([pad[i:i + F] for i in range(2 * ctx + 1)], 1)             # (F, 16*(2ctx+1))
    s = np.sqrt(np.mean(win ** 2, 1)) + 1e-9
    X = win / s[:, None]
    if hf_avail is not None:
        hp = np.pad(C[:, HF] * hf_avail[:, None], ((1, 1), (0, 0)))
        ap = np.pad(hf_avail.astype(float), (1, 1))
        prev, nxt = hp[:F] / s[:, None], hp[2:] / s[:, None]
        X = np.concatenate([X, prev * ap[:F, None], nxt * ap[2:, None], ap[:F, None], ap[2:, None]], 1)
    return X, s


def train_ridge(files, src, hf_mode, rng, seconds=40.0, lam=1e-2):
    Xs, Ys = [], []
    for f in files:
        import soundfile as sf
        info = sf.info(os.path.join(src, f))
        x, fs = sf.read(os.path.join(src, f), dtype="float64", always_2d=True, start=int(info.samplerate * 20), frames=int(info.samplerate * seconds))
        m = wavio.resample(x[:, :2], fs, FS).mean(1) * 0.5
        C = coefs_of(m)
        avail = None
        if hf_mode:
            avail = (rng.random(len(C)) > 0.2).astype(float)                       # 前後の高域が届いている割合 (80%)
        X, s = feats(C, avail)
        Y = C[:, HF] / s[:, None]
        keep = s > 1e-4                                                              # ほぼ無音のフレームは除く (±1 の実数の単位)
        Xs.append(X[keep]); Ys.append(Y[keep])
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    A = X.T @ X
    W = np.linalg.solve(A + lam * np.trace(A) / len(A) * np.eye(len(A)), X.T @ Y)
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--src", default=os.path.join(GRACEFM, "eval_audio", "train_lossless"))
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    from visqol import VisqolApi
    api = VisqolApi()
    api.create(mode="audio")
    rng = np.random.default_rng(0)
    files = sorted(f for f in os.listdir(a.src) if f.endswith(".flac"))
    files = [f for i, f in enumerate(files) if i % 7 != 3 and "Ambient" not in f]                # 検証用と ambient は学習に使わない
    print(f"学習音源 {len(files)} 本 (検証用・ambient を除く)。回帰を作る...", flush=True)
    W1 = train_ridge(files, a.src, False, rng)
    W2 = train_ridge(files, a.src, True, rng)
    sos = butter(4, 6000 / (FS / 2), output="sos")
    names = ["zero", "zero+IIR", "lmmse", "lmmse+IIR"]
    acc = {sc: {n: [] for n in names} for sc in ("S1", "S2")}
    spec = {sc: {n: [] for n in names} for sc in ("S1", "S2")}
    print(f"{'素材':10}{'条件':>4}" + "".join(f"{n:>22}" for n in names) + "   (高域SNR[dB] / 全体SNR[dB] / ViSQOL)")
    for mat in MATERIALS:
        x = load(mat, a.seconds)
        m = x.mean(1)
        m = np.asarray(wavio.to_pcm24(m[:, None])[:, 0], float) / (1 << 23)
        C = coefs_of(m)
        n = len(C) * 64
        ref = m[:n]
        ref48 = wavio.resample(ref, FS, 48000)
        for sc in ("S1", "S2"):
            erase = np.ones(len(C), bool) if sc == "S1" else rng.random(len(C)) < 0.2
            avail = None if sc == "S1" else (~erase).astype(float)
            X, s = feats(C, avail)
            W = W1 if sc == "S1" else W2
            Ce = C.copy()
            Ce[erase, 16:] = 0
            Cw = Ce.copy()
            Cw[erase, 16:] = (X @ W * s[:, None])[erase]
            row = {}
            for nm, Cx, iir in (("zero", Ce, False), ("zero+IIR", Ce, True), ("lmmse", Cw, False), ("lmmse+IIR", Cw, True)):
                y = haar_inv(Cx).reshape(-1)
                if iir:
                    yi = np.roll(sosfilt(sos, y), -2)                              # 常時計算。群遅延 約 2 サンプルを補償
                    g = np.repeat(erase.astype(float), 64)
                    g = np.convolve(g, np.ones(33) / 33, "same")                    # 約 1 ms のクロスフェード (消したフレームの前後)
                    y = g * yi + (1 - g) * y                                        # 消したフレームだけ IIR に切り替える
                hf_err = np.sum((Cx[erase][:, HF] - C[erase][:, HF]) ** 2)
                hf_snr = 10 * np.log10(np.sum(C[erase][:, HF] ** 2) / max(hf_err, 1e-30)) if not iir else float("nan")
                snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - y) ** 2), 1e-30))
                mos = float(api.measure_from_arrays(ref48, wavio.resample(y, FS, 48000), sample_rate=48000).moslqo)
                f, Pr = welch(ref, FS, nperseg=2048)
                _, Pe = welch(ref - y, FS, nperseg=2048)
                bs = [10 * np.log10(Pr[(f >= lo) & (f < hi)].sum() / max(Pe[(f >= lo) & (f < hi)].sum(), 1e-30)) for lo, hi in ((4000, 8000), (8000, 16000))]
                row[nm] = (hf_snr, snr, mos)
                acc[sc][nm].append((hf_snr, snr, mos))
                spec[sc][nm].append(bs)
            print(f"{mat:10}{sc:>4}" + "".join(f"{row[n][0]:7.2f}/{row[n][1]:5.1f}/{row[n][2]:5.2f}   " for n in names), flush=True)
    print("\n6 素材の平均")
    for sc in ("S1", "S2"):
        print(f"{sc:14}" + "".join(f"{np.nanmean([v[0] for v in acc[sc][n]]):7.2f}/{np.mean([v[1] for v in acc[sc][n]]):5.1f}/{np.mean([v[2] for v in acc[sc][n]]):5.2f}   " for n in names))
    print("\nスペクトルで見た高域の SNR [dB] (誤差 = 出力 - 原音。4-8 kHz / 8-16 kHz、全体の信号に対する帯域のエネルギー比)")
    for sc in ("S1", "S2"):
        print(f"{sc:6}" + "".join(f"{n:>12}: {np.mean([v[0] for v in spec[sc][n]]):6.2f} / {np.mean([v[1] for v in spec[sc][n]]):6.2f}   " for n in names))
    print("素材別 (S1、lmmse): " + ", ".join(f"{m} {spec['S1']['lmmse'][i][0]:.1f}/{spec['S1']['lmmse'][i][1]:.1f}" for i, m in enumerate(MATERIALS)))


if __name__ == "__main__":
    main()
