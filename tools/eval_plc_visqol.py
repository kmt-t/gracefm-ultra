#!/usr/bin/env python3
"""
欠落補間の聴感評価 (ViSQOL MOS-LQO) と全体 SNR。検証音源 (M 成分、32 kHz) に、長さ g フレームの欠落を一定間隔で入れ、
各手法で埋めた信号全体を ViSQOL (audio モード、48 kHz) で原音と比べる。値は音源の平均。

  .venv/bin/python tools/eval_plc_visqol.py --gaps 3,8,25 --ckpt runs/plc.pt,runs/plc2.pt
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_plc import BLOCK, FS, GRACEFM, _gfm, val_clips  # noqa: E402
from gracefm_ultra.plc_ar import interpolate  # noqa: E402
from gracefm_ultra.plcnet import load_predictor  # noqa: E402
from tools.common import wavio  # noqa: E402


def apply(x, g, period, fn):
    y = x.copy()
    for p0 in range(2048, len(x) - 2048 - g * BLOCK, period):
        y[p0:p0 + g * BLOCK] = fn(x, p0, p0 + g * BLOCK)
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(GRACEFM, "eval_audio", "train_lossless"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--gaps", default="3,8,25")
    ap.add_argument("--period-ms", type=float, default=300.0, help="欠落の間隔")
    ap.add_argument("--ckpt", default="", help="カンマ区切り")
    a = ap.parse_args()
    from visqol import VisqolApi
    api = VisqolApi()
    api.create(mode="audio")
    meths = {"zero": lambda x, p, q: np.zeros(q - p), "gfm": _gfm, "ar2": lambda x, p, q: interpolate(x, p, q)}
    for c in [c for c in a.ckpt.split(",") if c]:
        meths["net:" + os.path.basename(c)] = load_predictor(c)
    clips = val_clips(a.src, a.seconds, a.files)
    period = int(a.period_ms / 1000 * FS) // BLOCK * BLOCK
    print(f"{len(clips)} 音源 × {a.seconds}s、欠落は {a.period_ms:.0f} ms ごと。ViSQOL MOS-LQO (上) / 全体 SNR [dB] (下)\n")
    print(f"{'gap':>4}{'欠落率':>8}  " + "  ".join(f"{k:>14}" for k in meths))
    for g in [int(v) for v in a.gaps.split(",")]:
        mos = {k: [] for k in meths}
        snr = {k: [] for k in meths}
        for x in clips:
            ref48 = wavio.resample(x / (1 << 23), FS, 48000)
            for k, fn in meths.items():
                y = apply(x, g, period, fn)
                mos[k].append(float(api.measure_from_arrays(ref48, wavio.resample(y / (1 << 23), FS, 48000), sample_rate=48000).moslqo))
                snr[k].append(10 * np.log10(np.sum(x ** 2) / max(np.sum((x - y) ** 2), 1e-9)))
        loss = g * BLOCK / period * 100
        print(f"{g:4d}{loss:7.1f}%  " + "  ".join(f"{np.mean(mos[k]):14.3f}" for k in meths))
        print(f"{'':12}  " + "  ".join(f"{np.mean(snr[k]):14.2f}" for k in meths))


if __name__ == "__main__":
    main()
