#!/usr/bin/env python3
"""
量子化ノイズの高域への漏れの図 (通信路なし、B200)。素材を指定して、原音と、誤差 (出力 − 原音) のスペクトログラムと、
平均スペクトルを並べる。ピアノのように高域が空の素材では、漏れ雑音が高域の本物の音と同じレベルになる。

  .venv/bin/python tools/plot_leak.py --material piano --out runs/demo/leak_piano.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import spectrogram, welch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from materials import load  # noqa: E402
from gracefm_ultra.gfm import GRACEFM  # noqa: E402,F401
from gracefm.decoder import roundtrip  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default="piano")
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "demo", "leak_piano.png"))
    a = ap.parse_args()
    p = PROFILES[a.profile]
    x = load(a.material, a.seconds)
    xi = wavio.to_pcm24(x)
    y = roundtrip(xi, p).astype(float) / (1 << 23)
    ref = xi.astype(float) / (1 << 23)
    err = y - ref
    m_ref, m_err = ref.mean(1), err.mean(1)                                       # 左右の平均で描く
    fs = p.fs
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    vmin, vmax = -140, -40
    for a_, sig, title in ((ax[0, 0], m_ref, "① 原音のスペクトログラム"), (ax[0, 1], m_err, "② 量子化の誤差 (出力 − 原音) のスペクトログラム")):
        f, t, S = spectrogram(sig, fs, nperseg=1024, noverlap=768)
        im = a_.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-16), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
        a_.set_title(title, fontsize=13)
        a_.set_ylabel("周波数 [kHz]")
        a_.set_xlabel("時間 [秒]")
        a_.axhspan(8, 16, fill=False, edgecolor="cyan", lw=1.5, ls="--")
        fig.colorbar(im, ax=a_, label="レベル [dB] (明るいほど大きい)")
    f, Pr = welch(m_ref, fs, nperseg=4096)
    _, Pe = welch(m_err, fs, nperseg=4096)
    b = ax[1, 0]
    b.plot(f / 1000, 10 * np.log10(Pr + 1e-18), label="原音 (本物の音)", color="tab:blue", lw=1.5)
    b.plot(f / 1000, 10 * np.log10(Pe + 1e-18), label="量子化の誤差 (漏れ雑音)", color="tab:red", lw=1.5)
    b.axvspan(8, 16, color="cyan", alpha=0.12)
    b.set_title("③ 平均スペクトル: 高域 (8〜16kHz、水色) で、雑音が本物の音と並ぶ", fontsize=13)
    b.set_xlabel("周波数 [kHz]")
    b.set_ylabel("レベル [dB/Hz]")
    b.legend()
    b.grid(alpha=0.3)
    b.set_ylim(-150, -40)
    c = ax[1, 1]
    bands = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))
    diffs = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        diffs.append(10 * np.log10(Pe[m].sum() / Pr[m].sum()))
    labels = [f"{lo // 1000}-{hi // 1000}k" for lo, hi in bands]
    colors = ["tab:red" if d > -3 else "tab:green" for d in diffs]
    c.bar(labels, diffs, color=colors)
    c.axhline(0, color="k", lw=1)
    c.set_title("④ 雑音 − 本物の音 [dB]。0 に近いほど、雑音が隠れない (赤: -3dB より上)", fontsize=13)
    c.set_ylabel("雑音 − 本物の音 [dB]")
    c.set_xlabel("周波数帯 [Hz]")
    for i, d in enumerate(diffs):
        c.text(i, d + (1 if d >= 0 else -4), f"{d:+.1f}", ha="center", fontsize=11)
    c.set_ylim(-70, 10)
    fig.suptitle(f"{a.profile} の量子化のみ (通信路なし)。素材: {a.material}。低域の量子化雑音が、高域へ漏れる様子", fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=70)
    print("保存:", os.path.abspath(a.out), " 帯域ごとの 雑音−原音 [dB]:", [f"{d:+.1f}" for d in diffs])


if __name__ == "__main__":
    main()
