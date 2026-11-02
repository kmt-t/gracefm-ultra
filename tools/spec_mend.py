#!/usr/bin/env python3
"""
MendNet の補間音のノイズの調査。原音・補間後・誤差のスペクトログラム (使えない区間を帯で示す)、フレームごと・帯域ごとの誤差、
使えるフレームと使えないフレームの誤差の帯域別の内訳を出す。

  .venv/bin/python tools/spec_mend.py --ckpt runs/mend_B200_v2.pt --scen 3 --out runs/demo_mend/spec
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
from demo_mend import FS, mend_stream, to_audio  # noqa: E402
from gracefm_ultra.mendnet import BAND, MendNet, join_ar  # noqa: E402

BANDS_HZ = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]           # 帯域 k = 0〜4 の周波数範囲 (32 kHz、Haar の 5 帯域)


def spectrogram(x, n=512):
    w = np.hanning(n)
    hop = n // 4
    fr = np.stack([x[i:i + n] * w for i in range(0, len(x) - n, hop)])
    return 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-9).T, hop


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--file", type=int, default=0)
    ap.add_argument("--scen", type=int, default=3)
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--view", type=float, default=2.5, help="スペクトログラムに出す長さ [秒]")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = sorted(glob.glob(os.path.join(a.data, a.profile, "mend", "val", "*.npz")))[a.file]
    z = np.load(f)
    r = slice(int(a.start * FS / 64), int((a.start + a.seconds) * FS / 64))
    truth, dec, var = z["truth"][r], z["dec"][a.scen][r], z["var"][a.scen][r]
    st, gap = z["status"][a.scen][r], z["gap"][a.scen][r]
    ck = torch.load(a.ckpt, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    us = st <= 1
    out = []
    for ch in range(2):
        s = slice(ch * 64, ch * 64 + 64)
        hint = (z["hint_h"][a.scen][r][:, s], z["hint_q"][a.scen][r][:, ch]) if ck.get("hint", False) else None
        out.append(join_ar(mend_stream(net, dec[:, s], var[:, ch], st, gap, ch, dev, hint, ck.get("refine", False)), us))
    sets = {"original": to_audio(truth[:, :64], truth[:, 64:]), "mendnet_join": to_audio(*out)}
    sets["error"] = sets["mendnet_join"] - sets["original"]
    sets["received"] = to_audio(dec[:, :64], dec[:, 64:])

    # フレーム × 帯域の誤差 (M と S の合計、係数領域 = 時間領域)
    err = np.concatenate([out[0] - truth[:, :64], out[1] - truth[:, 64:]], 1)          # (T,128)
    sig = np.concatenate([truth[:, :64], truth[:, 64:]], 1)
    band2 = np.concatenate([BAND, BAND])
    e_fb = np.stack([(err[:, band2 == k] ** 2).sum(1) for k in range(5)], 1)
    s_fb = np.stack([(sig[:, band2 == k] ** 2).sum(1) for k in range(5)], 1)
    print(f"シナリオ {z['scen'][a.scen]}  使えないフレーム {int((~us).sum())}/{len(us)}")
    print("帯域 周波数       信号の電力[dB]  誤差/信号[dB]: 使えるフレーム  使えないフレーム   全誤差のうち使えないフレームの割合")
    for k in range(5):
        tot = e_fb[:, k].sum()
        print(
            f"k={k} {BANDS_HZ[k][0]:5d}-{BANDS_HZ[k][1]:5d}Hz "
            f"{10 * np.log10(s_fb[:, k].mean() + 1e-30):8.1f}   "
            f"{10 * np.log10(e_fb[us, k].sum() / s_fb[us, k].sum()):8.1f}   {10 * np.log10(e_fb[~us, k].sum() / s_fb[~us, k].sum()):8.1f}   "
            f"{100 * e_fb[~us, k].sum() / tot:6.1f}%"
        )

    t0 = int(a.view * FS / 64)
    fig, ax = plt.subplots(6, 1, figsize=(14, 17), sharex=False)
    names = ["original", "received", "mendnet_join", "error"]
    for i, n in enumerate(names):
        sp, hop = spectrogram(sets[n][:t0 * 64, 0])
        im = ax[i].imshow(sp, origin="lower", aspect="auto", extent=[0, a.view, 0, FS / 2000], vmin=-20 if n != "error" else -40, vmax=60 if n != "error" else 40, cmap="magma")
        ax[i].set_title(f"{n} (L), cyan = unusable frames")
        ax[i].set_ylabel("kHz")
        for j in np.nonzero(~us[:t0])[0]:
            ax[i].axvspan(j * 64 / FS, (j + 1) * 64 / FS, color="cyan", alpha=0.25, lw=0)
        fig.colorbar(im, ax=ax[i], pad=0.01, label="dB")
    tt = np.arange(t0) * 64 / FS
    for k in range(5):
        ax[4].plot(tt, 10 * np.log10(e_fb[:t0, k] + 1e-14), lw=0.8, label=f"k={k}")
    ax[4].set_title("error power per frame [dB] by band")
    ax[4].legend(ncol=5)
    for k in range(5):
        ax[5].plot(tt, 10 * np.log10(e_fb[:t0, k] / (s_fb[:t0, k] + 1e-14) + 1e-9), lw=0.8, label=f"k={k}")
    ax[5].set_title("error / signal per frame [dB] by band")
    for j in np.nonzero(~us[:t0])[0]:
        for x in (ax[4], ax[5]):
            x.axvspan(j * 64 / FS, (j + 1) * 64 / FS, color="cyan", alpha=0.2, lw=0)
    ax[5].set_xlabel("s")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"spec_s{a.scen}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    print(path)


if __name__ == "__main__":
    main()
