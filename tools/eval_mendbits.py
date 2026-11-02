#!/usr/bin/env python3
"""
MendNet (ビット出力版) の評価。検証データの窓で、フレームの状態 (使える・壊れた・欠落) ごとに、受信 LLR のまま (基準) と比べる。
- 係数の SNR [dB]: ビットの確率から作る仮数の期待値と真の仮数の差 (Σ (p1 - b) 2^pos) × 刻み を、原音の係数 (真の刻みを使う) と比べる。音声としての誤り
- 仮数の上位ビット (最上位から 1〜3 番目) のビット誤り率
- 刻み (ヘッダ): log2 の誤差の絶対値の平均と、指数が合う割合

  .venv/bin/python tools/eval_mendbits.py --ckpt runs/mendbits_B200.pt --windows 1500
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm_ultra.mendnet import STEP_OFFSET, MendNet, bits_out, build_input_bits, step_feature  # noqa: E402
from train_mend import Windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--windows", type=int, default=1500)
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    va = Windows(sorted(glob.glob(os.path.join(a.data, a.profile, "mend", "val", "*.npz"))), dev, bits=True)
    ck = torch.load(a.ckpt, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    rng = np.random.default_rng(5)
    names = ["使える", "壊れた", "欠落"]
    models = ("受信LLR", "MendNet")
    top = 3
    sq = {m: {n: [0.0, 0.0] for n in names} for m in models}                          # 誤差の二乗和、原音の二乗和
    msb = {m: {n: np.zeros((top, 2)) for n in names} for m in models}                 # 上位 top 個のビット: 誤り数、ビット数
    stp = {n: np.zeros(5) for n in names}                                             # 刻みの誤差、|誤差| < 0.5、< 1.5 の数、係数数、予測の刻みを使った誤差の二乗和
    tsum = {n: 0.0 for n in names}
    for _ in range(a.windows // 25):
        b = va.batch(25, rng, True)
        with torch.no_grad():
            x, _, usable, sf = build_input_bits(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], b["hint_h"], b["hint_q"], b["llr"], b["dstep"])
            out = bits_out(net(x), b["llr"], sf, b["dstep"])
        raw = (b["llr"].float() / 4, sf)
        sf_true, _ = step_feature(b["step"])
        cls = b["status"].long().clamp(1, 3) - 1                                      # 0 = 使える (0, 1)、1 = 壊れた、2 = 欠落
        cls = torch.where(usable, torch.zeros_like(cls), cls)
        ok = b["ok"].expand_as(b["bits"]).bool()
        w = (2.0 ** b["pos"].float())[:, None, :] * ok                                # (B,1,NB)
        rank = b["pos"].max(1, keepdim=True).values - b["pos"]                       # 0 = 最上位
        for m, (llr, st) in zip(models, (raw, out)):
            diff = (torch.sigmoid(-llr) - b["bits"].float()) * w                      # 仮数の誤差への寄与
            e = torch.zeros(b["bits"].shape[0], b["bits"].shape[1], 64, device=dev).scatter_add_(2, b["coef_of_bit"][:, None, :].expand_as(diff), diff)
            e = e * b["step"] / 2 ** 22
            err = (llr < 0) != (b["bits"] == 1)
            for c, n in enumerate(names):
                sel = cls == c
                sq[m][n][0] += (e[sel] ** 2).sum().item()
                sq[m][n][1] += (b["target"][sel] ** 2).sum().item()
                for r in range(top):
                    s2 = sel[:, :, None] & ok & (rank == r)[:, None, :]
                    msb[m][n][r] += [err[s2].sum().item(), s2.sum().item()]
            if m == "MendNet":
                e_mend = e
                d = ((st - sf_true) * STEP_OFFSET).abs()
                for c, n in enumerate(names):
                    sel = (cls == c)[:, :, None].expand_as(d)
                    stp[n] += [d[sel].sum().item(), (d[sel] < 0.5).sum().item(), (d[sel] < 1.5).sum().item(), sel.sum().item(), 0.0]
                    r = 2.0 ** ((st - sf_true) * STEP_OFFSET)                                 # 予測の刻み / 真の刻み
                    t = b["target"]
                    stp[n][4] += (((t * (r - 1)) + e_mend * r)[cls == c] ** 2).sum().item()
                    tsum[n] += (t[cls == c] ** 2).sum().item()
    for m in models:
        print(m)
        for n in names:
            e, t = sq[m][n]
            r = " ".join(f"{msb[m][n][k, 0] / max(msb[m][n][k, 1], 1):.4f}" for k in range(top))
            print(f"  {n:4s} 係数 SNR {10 * np.log10(t / max(e, 1e-30)):+6.2f} dB  上位ビット 1〜{top} 番目の誤り率 {r}")
    print("刻み (MendNet): log2 の誤差の平均 / 指数が合う割合 / ±1 以内の割合 / 予測した刻みを使った係数 SNR")
    for n in names:
        d, e0, e1, c, q = stp[n]
        print(f"  {n:4s} {d / max(c, 1):.3f} / {e0 / max(c, 1):.3f} / {e1 / max(c, 1):.3f} / {10 * np.log10(tsum[n] / max(q, 1e-30)):+6.2f} dB")


if __name__ == "__main__":
    main()
