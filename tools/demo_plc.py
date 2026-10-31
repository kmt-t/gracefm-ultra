#!/usr/bin/env python3
"""
欠落補間のデモ音声。ピアノ (Emotional_piano.wav) の M 成分 (32 kHz) に、長さ g フレームの欠落を一定間隔で入れ、
各手法で埋めた音を runs/demo/ に WAV (32 kHz、24 bit) で書き出し、SNR と ViSQOL を出す。

  .venv/bin/python tools/demo_plc.py --gaps 8,25 --ckpt runs/plc2.pt
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_plc import BLOCK, FS, GRACEFM, _gfm  # noqa: E402
from eval_plc_visqol import apply  # noqa: E402
from gracefm_ultra.plc_ar import interpolate  # noqa: E402
from gracefm_ultra.plcnet import load_predictor  # noqa: E402
from tools.common import wavio  # noqa: E402


def lowpass_gap(fn, fc, pad=512):
    """欠落区間の補間 fn に、ゼロ位相のローパス (fc [Hz]) をかける。前後の既知部も含めて濾波し、端の影響を避ける"""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, fc / (FS / 2), output="sos")

    def g(x, p, q):
        est = fn(x, p, q)
        seg = np.array(x[p - pad:q + pad], float)
        seg[pad:pad + (q - p)] = est
        return sosfiltfilt(sos, seg)[pad:pad + (q - p)]
    return g


def click_db(ref, y, g, period):
    """欠落の両端 (±8 サンプル) の 2 階差分のエネルギーを、原音の同じ位置に対する比 [dB] で。大きいほど境界の不連続 (ざっ) が強い"""
    num = den = 0.0
    for p0 in range(2048, len(ref) - 2048 - g * BLOCK, period):
        for e in (p0, p0 + g * BLOCK):
            sl = slice(e - 8, e + 8)
            num += float(np.sum(np.diff(y[sl], 2) ** 2))
            den += float(np.sum(np.diff(ref[sl], 2) ** 2))
    return 10 * np.log10(num / max(den, 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default=os.path.join(GRACEFM, "eval_audio", "lossless", "Emotional_piano.wav"))
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--gaps", default="8,25")
    ap.add_argument("--period-ms", type=float, default=300.0)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "runs", "plc2.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "demo"))
    a = ap.parse_args()
    from visqol import VisqolApi
    api = VisqolApi()
    api.create(mode="audio")
    x, fs = wavio.read(a.audio)
    x = wavio.resample(x, fs, FS)
    m = x.mean(1)[int(a.start * FS):int((a.start + a.seconds) * FS)]
    m = m / np.abs(m).max() * 0.9                                           # 聴きやすいように音量をそろえる
    ref = wavio.to_pcm24(m[:, None])[:, 0].astype(float)
    os.makedirs(a.out, exist_ok=True)
    wavio.write_pcm24(os.path.join(a.out, "piano_original.wav"), ref / (1 << 23), FS)
    meths = {"zero": lambda x_, p, q: np.zeros(q - p), "gfm": _gfm, "ar2": lambda x_, p, q: interpolate(x_, p, q),
             "net_noxf": load_predictor(a.ckpt, xfade=0), "net": load_predictor(a.ckpt)}
    for fc in (6000, 3000):
        meths[f"net_lp{fc // 1000}k"] = lowpass_gap(load_predictor(a.ckpt), fc)
    period = int(a.period_ms / 1000 * FS) // BLOCK * BLOCK
    ref48 = wavio.resample(ref / (1 << 23), FS, 48000)
    print(f"ピアノ {a.seconds}s (32 kHz モノラル)、欠落は {a.period_ms:.0f} ms ごと。出力: {os.path.abspath(a.out)}\n")
    print(f"{'欠落':>10}{'手法':>10}{'SNR[dB]':>10}{'ViSQOL':>9}{'境界の不連続[dB]':>16}  ファイル")
    for g in [int(v) for v in a.gaps.split(",")]:
        for k, fn in meths.items():
            y = apply(ref, g, period, fn)
            snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - y) ** 2), 1e-9))
            mos = float(api.measure_from_arrays(ref48, wavio.resample(y / (1 << 23), FS, 48000), sample_rate=48000).moslqo)
            name = f"piano_gap{g}_{k}.wav"
            wavio.write_pcm24(os.path.join(a.out, name), np.clip(y / (1 << 23), -1, 1), FS)
            print(f"{g:3d}フレーム{k:>10}{snr:10.2f}{mos:9.3f}{click_db(ref, y, g, period):16.1f}  {name}")


if __name__ == "__main__":
    main()
