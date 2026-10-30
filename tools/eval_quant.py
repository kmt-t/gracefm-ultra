#!/usr/bin/env python3
"""
領域A (C/N が良く、ビット誤りがない): 量子化雑音だけが残る状況で、推定器がどこまで SNR を上げるか。
チャネルを通さず、符号化 → 逆量子化した係数 y に推定器を当て、量子化前の係数 t との誤差を帯域ごとに見る。

  E0        逆量子化そのまま (現行)
  W-band    ウィナー: σx² は帯域・±W フレームの平均 (平均 y² - Δ²/12)
  W-ctx     ウィナー: σx² を係数の周辺 (時間 ±W × 周波数 ±1) の平均から。係数ごとに利得が変わる
  C-ctx     センタロイド: Laplace 事前 (尺度 b は周辺の平均 |y|) での量子化区間の重心。0 側へ寄る
  Oracle    W-ctx の σx² を正解 t から求める (文脈の事前の上限の目安)
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.rx import GRACEFM  # noqa: E402,F401
from gracefm.codec import FRAC, attenuator_expanded, encode_block, exp_range, haar_forward  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402


def collect(p, x):
    """x: (n, 2) PCM24 → y, t, step (F, 64) と、係数の帯域番号 (64,)"""
    b = p.block
    nfr = len(x) // b
    y = np.zeros((nfr, b))
    t = np.zeros((nfr, b))
    step = np.ones((nfr, b))
    from gracefm.codec import dequantize
    band = np.concatenate([[k] * p.ncoef(k) for k in range(p.nbands)])
    for i in range(nfr):
        blk = x[i * b:(i + 1) * b]
        bd = encode_block(blk, p)
        m = (blk[:, 0] + blk[:, 1] + 2) >> 2
        t[i] = np.concatenate(haar_forward(m, p.levels))
        o = 0
        for k, bits in enumerate(p.bits_m):
            n = p.ncoef(k)
            if bits:
                y[i, o:o + n] = dequantize(bd.mant["M"][k], bd.eidx["M"][k], k, bits, p, "M", bd.attn["M"][k])
                a = attenuator_expanded(bd.attn["M"][k], p.attenuator_group("M", k), n)
                step[i, o:o + n] = 2.0 ** (int(bd.eidx["M"][k]) + exp_range(k, bits, p)[0] + FRAC - np.asarray(a, float))
            o += n
    return y, t, step, band


def ctx_mean(v, band, w, nf=3):
    """v: (F, 64)。各帯域の中で、時間 ±w × 周波数 ±(nf//2) の平均 (端は最近傍で延長)"""
    out = np.empty_like(v)
    for k in np.unique(band):
        sel = band == k
        out[:, sel] = uniform_filter(v[:, sel], size=(2 * w + 1, min(nf, sel.sum())), mode="nearest")
    return out


def estimators(y, t, step, band, w):
    q = step ** 2 / 12
    res = {"E0": y}
    sb = np.maximum(ctx_mean(y ** 2, band, w, nf=64) - q, 1e-3)               # 帯域全体 (周波数方向に全部)
    res["W-band"] = sb / (sb + q) * y
    sc = np.maximum(ctx_mean(y ** 2, band, w) - q, 1e-3)
    res["W-ctx"] = sc / (sc + q) * y
    b = np.maximum(ctx_mean(np.abs(y), band, w), 1e-3 * step)                 # Laplace の尺度
    z = step / (2 * b)
    off = b - step / 2 / np.tanh(np.minimum(z, 50))                         # 区間の重心の 0 側へのずれ (m≠0)
    cen = y - np.sign(y) * np.where(np.abs(y) > 0, -off, 0)                  # 重心 = |y| + off (off<0 なので 0 側へ)
    res["C-ctx"] = np.where(np.abs(y) > 0, cen, 0.0)
    so = np.maximum(ctx_mean(t ** 2, band, w), 1e-3)
    res["Oracle"] = so / (so + q) * y
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M")
    ap.add_argument("--src", default=os.path.join(os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm")), "eval_audio", "train_lossless"))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--w", type=int, default=2)
    a = ap.parse_args()
    p = PROFILES[a.profile]
    files = sorted(f for f in os.listdir(a.src) if f.endswith((".flac", ".wav")))
    files = [f for i, f in enumerate(files) if i % 7 == 3][:a.files]      # 検証用に分けた音源
    sig = np.zeros(p.nbands)
    err = {}
    for f in files:
        x, fs = wavio.read(os.path.join(a.src, f))
        x = wavio.to_pcm24(wavio.resample(x, fs, p.fs)[:int(a.seconds * p.fs)] * 0.5)
        y, t, step, band = collect(p, x)
        for k in range(p.nbands):
            sig[k] += (t[:, band == k] ** 2).sum()
        for name, e in estimators(y, t, step, band, a.w).items():
            err.setdefault(name, np.zeros(p.nbands))
            for k in range(p.nbands):
                err[name][k] += ((e[:, band == k] - t[:, band == k]) ** 2).sum()
    edges = ["0-1k", "1-2k", "2-4k", "4-8k", "8-16k"]
    print(f"{len(files)} 音源 × {a.seconds}s、{a.profile}、M成分。帯域別 SNR [dB] (窓 ±{a.w} フレーム)\n")
    print(f"{'':8}" + "".join(f"{e:>9}" for e in edges) + f"{'全体':>9}")
    for name, e in err.items():
        snr = 10 * np.log10(sig / e)
        print(f"{name:8}" + "".join(f"{v:9.2f}" for v in snr) + f"{10 * np.log10(sig.sum() / e.sum()):9.2f}")
    print("\n(参考) 帯域のエネルギー比 [dB, 全体比]: " + " ".join(f"{10 * np.log10(s / sig.sum()):6.1f}" for s in sig))


if __name__ == "__main__":
    main()
