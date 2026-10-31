#!/usr/bin/env python3
"""
量子化のみ (通信路なし) の B200 M 成分で、k=0〜3 (自由) の係数を量子化の区間 (復号値 ± 刻み/2) の中で動かして、
再構成波形の高域 (カットオフ以上) のエネルギーが最小になる値を最小二乗 (箱制約) で探す。k=4 は復号値のまま固定 (--free-k4 で k=4 も動かす)。
層間の関係 (本物の高域は小さい) から、k=0〜3 の量子化誤差が分かるかの確認。DNN なし。
  誤差 (出力 − 原音) の帯域別パワーを、復号値のまま (E0) と箱制約最小二乗の後で比べる。原音の帯域パワーも出す。

  .venv/bin/python tools/exp_cell_recon.py --frames 64 --chunks 6 --cutoff 4000
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import firwin

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.haarnet import haar_matrix  # noqa: E402
from gracefm_ultra.rx import GRACEFM  # noqa: E402,F401

FS = 32000
BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))


def synth(c, H):
    """c: (N, 64) → 波形 (N*64,)"""
    return (c @ H).reshape(-1)


def band_db(x, taps):
    """x の帯域別パワー [dB]。両端の taps 分は除く"""
    x = x[taps:-taps] * np.hanning(len(x) - 2 * taps)
    P = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return np.array([10 * np.log10(P[(f >= lo) & (f < hi)].sum() + 1e-9) for lo, hi in BANDS])


def material_coefs(name, seconds, profile):
    """素材を符号化して、M 成分の (量子化前の係数、復号値、刻み) を作る (gen_dataset と同じ手順)"""
    from exp_wiener_post import load
    from gracefm.codec import decode_coeffs, encode_block, haar_forward
    from gracefm.layout import Layout
    from gracefm.profiles import PROFILES
    from prep_haarnet import steps_of
    from tools.common import wavio
    p = PROFILES[profile]
    xs = wavio.to_pcm24(load(name, seconds))
    nfr = len(xs) // p.block
    c0, cq = np.zeros((nfr, 64)), np.zeros((nfr, 64))
    for i in range(nfr):
        blk = xs[i * p.block:(i + 1) * p.block]
        bd = encode_block(blk, p)
        c0[i] = np.concatenate(haar_forward((blk[:, 0] + blk[:, 1] + 2) >> 2, p.levels))
        cq[i] = np.concatenate(decode_coeffs(bd, p)["M"])
    st = steps_of(p, Layout(p), xs, nfr)[:, :64]
    return c0, cq, st.astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--cutoff", type=float, default=4000)
    ap.add_argument("--ntaps", type=int, default=129)
    ap.add_argument("--material", default="", help="素材名 (exp_wiener_post.MATERIALS)。指定すると val の npz の代わりに、その素材を符号化して使う")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--free-k4", action="store_true")
    ap.add_argument("--prior", action="store_true", help="事前分布つき (MAP): 高域のエネルギーを原音の高域レベルで、δ を一様分布の分散で正規化して足す。高域レベルは窓ごとの原音 (オラクル)")
    a = ap.parse_args()
    os.sched_setaffinity(0, {7})
    H = haar_matrix()
    N, nt = a.frames, a.ntaps
    hp = -firwin(nt, a.cutoff, fs=FS)
    hp[nt // 2] += 1.0                                    # 高域通過 = δ − 低域通過
    nfree = 64 if a.free_k4 else 32
    print(f"{a.profile} M 成分、{N} フレーム窓 {a.chunks} 個、高域 {a.cutoff:.0f} Hz 以上のエネルギーを最小化、自由な係数 {nfree}/64 個/フレーム\n")
    print(f"{'素材':26}" + "".join(f"{lo // 1000}-{hi // 1000}kHz".rjust(9) for lo, hi in BANDS) + "   (誤差パワー − 原音パワー [dB])")
    if a.material:
        srcs = [(a.material, lambda: material_coefs(a.material, a.seconds, a.profile))]
    else:
        def val(f):
            d = np.load(f)
            return d["coef0"][:, :64].astype(float), d["coefq"][:, :64].astype(float), d["step"][:, :64].astype(float)
        srcs = [(os.path.basename(f), (lambda f=f: val(f))) for f in sorted(glob.glob(os.path.join(HERE, "..", "data", a.profile, "hn", "val", "*.npz")))]
    for nm, getter in srcs:
        c0, cq, st = getter()
        e0 = np.linalg.norm(c0[:, :4], axis=1)
        starts = np.linspace(200, len(c0) - N - 200, a.chunks * 3).astype(int)
        starts = [s for s in starts if e0[s:s + N].mean() > 20 * st[s:s + N, :4].mean()][:a.chunks]
        acc = {k: np.zeros(len(BANDS)) for k in ("ref", "e0", "opt")}
        for s0 in starts:
            ct, cqq, step = c0[s0:s0 + N], cq[s0:s0 + N], st[s0:s0 + N]
            xt, xq = synth(ct, H), synth(cqq, H)
            # 自由な係数の単位応答 → 高域フィルタ後 (valid 畳み込み) の行列
            cols = []
            for n in range(N):
                for j in range(nfree):
                    u = np.zeros(N * 64)
                    u[n * 64:(n + 1) * 64] = H[j]
                    cols.append(np.convolve(u, hp, "valid"))
            G = np.array(cols).T
            b = -np.convolve(xq, hp, "valid")
            lo = (-0.5 * step[:, :nfree]).reshape(-1)
            if a.prior:
                sg = np.sqrt(np.mean(np.convolve(xt, hp, "valid") ** 2)) + 1e-9
                G, b = np.vstack([G / sg, np.diag(np.sqrt(12.0) / (step[:, :nfree].reshape(-1)))]), np.concatenate([b / sg, np.zeros(len(lo))])
            sol = lsq_linear(G, b, bounds=(lo, -lo), method="bvls" if G.shape[1] <= 1500 else "trf", lsmr_tol="auto", max_iter=200)
            c2 = cqq.copy()
            c2[:, :nfree] += sol.x.reshape(N, nfree)
            x2 = synth(c2, H)
            t = nt
            acc["ref"] += 10 ** (band_db(xt, t) / 10)
            acc["e0"] += 10 ** (band_db(xq - xt, t) / 10)
            acc["opt"] += 10 ** (band_db(x2 - xt, t) / 10)
        db = {k: 10 * np.log10(v / max(len(starts), 1)) for k, v in acc.items()}
        name = nm[:24]
        print(f"{name:26}" + "".join(f"{v:9.1f}" for v in db["ref"]) + "   原音パワー")
        print(f"{'':26}" + "".join(f"{db['e0'][i] - db['ref'][i]:+9.1f}" for i in range(len(BANDS))) + "   E0 (復号値のまま)")
        print(f"{'':26}" + "".join(f"{db['opt'][i] - db['ref'][i]:+9.1f}" for i in range(len(BANDS))) + "   箱制約最小二乗の後")
        e_all = lambda k: 10 * np.log10(sum(10 ** (db[k][i] / 10) for i in range(len(BANDS))))  # noqa: E731
        print(f"{'':26}  全体 SNR: E0 {e_all('ref') - e_all('e0'):.1f} dB → {e_all('ref') - e_all('opt'):.1f} dB\n")


if __name__ == "__main__":
    main()
