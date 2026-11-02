#!/usr/bin/env python3
"""
HaarNet-lite の評価 (検証データ)。生LLRの事後平均 E0 と比べる。欠落フレームを除く。
通信路 (AWGN/Rayleigh) と Es/N0 の帯域ごとに、誤差エネルギーを合算して SNR を出す。
列: 全体 (M と S の係数エネルギーの合計 = L/R の SNR と同じ尺度)、M の k=0 (最低域)、S の k=0。

  .venv/bin/python tools/eval_haarnet.py --profile B200 --ckpt runs/baseline/B200.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm.layout import Layout  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from hn_index import dataset_index  # noqa: E402
from gracefm_ultra.haarnet import WIN, HaarNetLite, features, haar_matrix, make_meta  # noqa: E402

HM = torch.from_numpy(haar_matrix()).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--frames", type=int, default=10000)
    ap.add_argument("--sub", default="hn_big", help="評価データの置き場 data/<profile>/<sub>/val")
    a = ap.parse_args()
    ckpt = a.ckpt or os.path.join(HERE, "..", "runs", f"hn_{a.profile}.pt")
    pin()
    torch.set_num_threads(min(len(CPUS), 4))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu")
    net = HaarNetLite(**ck["cfg"]).eval().to(dev)
    net.load_state_dict(ck["model"])
    p = PROFILES[a.profile]
    rows = []
    edge = 8
    for f in sorted(glob.glob(os.path.join(HERE, "..", "data", a.profile, a.sub, "val", "*.npz"))):
        d = np.load(f)
        meta = make_meta(d, p).to(dev)
        cidx = torch.from_numpy(dataset_index(p, Layout(p))[0]).to(dev)
        group = d["group"]
        g0m = group == 0
        g0s = (group == int(np.max(group[:64]) + 1)) if meta.ncoef > 64 else np.zeros(meta.ncoef, bool)   # S の k=0 (M の群の次)
        F = min(a.frames, d["llr"].shape[1])
        for s in range(d["llr"].shape[0]):
            acc = np.zeros(7)           # 信号(全体), E0誤差(全体), net誤差(全体), 信号(M k0), E0(M k0), net(M k0), ...
            lk = np.zeros((2, 3))       # 漏れ: 帯域 (4-8k, 8-16k) ごとの [原音, E0誤差, net誤差] のパワー (M 成分、欠落のない窓だけ)
            accs = np.zeros(3)          # S の k=0: 信号, E0, net
            starts = list(range(0, F - WIN + 1, WIN - 2 * edge))
            for b0 in range(0, len(starts), 16):
                ss = starts[b0:b0 + 16]
                cut = lambda arr: torch.from_numpy(np.stack([arr[t:t + WIN] for t in ss])).to(dev)  # noqa: E731
                llr, var = cut(d["llr"][s].astype(np.float32)), cut(d["var"][s].astype(np.float32))
                lost, step, truth = cut(d["lost"][s]), cut(d["step"]), cut(d["coef0"])
                with torch.no_grad():
                    x, S, e0n = features(llr, step, var, lost, meta)
                    est = (e0n + net(x)) * S
                    e0 = e0n * S
                keep = (~lost).float()[..., None].clone()
                keep[:, :edge] = 0
                keep[:, -edge:] = 0
                sq = lambda arr, m=None: float((arr ** 2 * keep * (1 if m is None else torch.from_numpy(m).float().to(dev))).sum())  # noqa: E731
                acc[:3] += [sq(truth), sq(e0 - truth), sq(est - truth)]
                acc[3:6] += [sq(truth, g0m), sq(e0 - truth, g0m), sq(est - truth, g0m)]
                if g0s.any():
                    accs += [sq(truth, g0s), sq(e0 - truth, g0s), sq(est - truth, g0s)]
                Hm = HM.to(dev)
                def wave(c):                                                                # M 成分の時間波形 (端を除く)。ない帯域は 0 で埋める
                    full = torch.zeros(c.shape[0], c.shape[1] - 2 * edge, 128, device=dev)
                    full[..., cidx] = c[:, edge:-edge]
                    return (full[..., :64] @ Hm).reshape(c.shape[0], -1)
                wr, w0, w1 = wave(truth), wave(e0 - truth), wave(est - truth)
                okw = (~lost[:, edge:-edge].any(1)).float()[:, None]
                win = torch.hann_window(wr.shape[1], device=dev)
                fr = np.fft.rfftfreq(wr.shape[1], 1 / 32000)
                for bi, (flo, fhi) in enumerate(((4000, 8000), (8000, 16000))):
                    m = torch.from_numpy((fr >= flo) & (fr < fhi)).to(dev)
                    for k_, w_ in enumerate((wr, w0, w1)):
                        P = (torch.fft.rfft(w_ * win).abs() ** 2)[:, m].sum(1)
                        lk[bi, k_] += float((P * okw[:, 0]).sum())
            rows.append((d["scen"][s][0], d["scen"][s][1], *acc[:6], *accs, *lk.reshape(-1)))
    r = np.array(rows)
    snr = lambda s_, e: 10 * np.log10(s_ / np.maximum(e, 1e-30))  # noqa: E731
    leaks = []
    print(f"{a.profile}: SNR [dB]。欠落フレームを除く全フレーム (CRC 失敗フレームも含む)。各列は E0 → HaarNet-lite")
    print("                         n        全体(M+S)            M k=0 (0-1kHz)          S k=0")
    for fade in (0, 1):
        for lo, hi in ((0, 6), (6, 10), (10, 14), (14, 20), (20, 28), (28, 41)):
            m = (r[:, 1] == fade) & (r[:, 0] >= lo) & (r[:, 0] < hi)
            if not m.any():
                continue
            c = r[m].sum(0)
            line = f"{'rayleigh' if fade else 'awgn    '} {lo:2d}-{hi:2d} dB {m.sum():3d}   "
            line += f"{snr(c[2], c[3]):6.2f} → {snr(c[2], c[4]):6.2f}    {snr(c[5], c[6]):6.2f} → {snr(c[5], c[7]):6.2f}   "
            line += f"{snr(c[8], c[9]):6.2f} → {snr(c[8], c[10]):6.2f}" if c[8] > 0 else "   --"
            print(line)
            leaks.append((fade, lo, hi, c[11:17]))
    print("\n高域の漏れ: 誤差 − 原音 [dB] (M 成分、欠落のない窓、4-8 kHz / 8-16 kHz)。正なら誤差の方が原音より大きい。E0 → HaarNet-lite")
    for fade, lo, hi, v in leaks:
        d = lambda i: 10 * np.log10(max(v[i + 1], 1e-30) / max(v[i], 1e-30))  # noqa: E731
        print(f"{'rayleigh' if fade else 'awgn    '} {lo:2d}-{hi:2d} dB   4-8k: {d(0):+6.1f} → {d(1) if False else 10 * np.log10(max(v[2], 1e-30) / max(v[0], 1e-30)):+6.1f}    8-16k: {10 * np.log10(max(v[4], 1e-30) / max(v[3], 1e-30)):+6.1f} → {10 * np.log10(max(v[5], 1e-30) / max(v[3], 1e-30)):+6.1f}")


if __name__ == "__main__":
    main()

