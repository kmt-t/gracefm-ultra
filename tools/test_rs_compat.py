#!/usr/bin/env python3
"""
Rust の送信（engine/gfm-tx、浮動小数点）が、本家 GraceFM（Python）の受信で正しく復号できること（ビット列の形式の準拠）を確かめる。
Rust が出したシンボルを、ほぼ雑音なしの通信路（Es/N0 = 40 dB）に通して、本家の受信にかける:
  - 基本 CRC・高域 CRC の不一致、欠落が 0
  - 本家の受信（デコーダだけ）の原音との SNR が、本家のエンコーダ → 本家のデコーダでの値（下の BASELINE_SNR、2 秒・同じ音源の区間）から 0.5 dB 以内

  .venv/bin/python tools/test_rs_compat.py
"""
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.gfm import GRACEFM  # noqa: E402
from gracefm.decoder import Decoder  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402
import rs  # noqa: E402

SRC = os.path.join(GRACEFM, "eval_audio", "train_lossless")
# 本家のエンコーダ → 本家のデコーダでの SNR [dB]（この音源の区間、2 秒）。GraceFM 8ba0ae0 で測定
BASELINE_SNR = {"B200": 55.77, "B200M": 12.49, "M700": 56.77, "M700M": 12.49, "SUB100": 22.83, "SUB50M": 12.12, "SUB100-48K": 25.37,
                "B200-L": 57.95, "B200M-L": 12.49, "M700-L": 59.31, "M700M-L": 12.49, "SUB100-L": 25.77, "SUB50M-L": 12.29, "SUB100-48K-L": 29.40}
TOLERANCE_DB = 0.5


def clip(p):
    """試験用の音源の区間（2 秒）→ 符号つき 24 bit 整数 (n, 2)"""
    files = sorted(f for f in os.listdir(SRC) if f.endswith((".flac", ".wav")))
    x, fs = sf.read(os.path.join(SRC, files[3]), dtype="float64", always_2d=True, start=int(10 * 44100), frames=int(3.5 * 44100))
    xs = wavio.resample(x[:, :2], fs, p.fs)[int(0.5 * p.fs):int(0.5 * p.fs) + int(2 * p.fs)] * 0.5
    xi = wavio.to_pcm24(xs)
    return xi[:len(xi) // p.block * p.block]


def snr(ref, y):
    return 10 * np.log10((ref[128:] ** 2).sum() / ((y[128:] - ref[128:]) ** 2).sum())


def main():
    bad = 0
    for name, base in BASELINE_SNR.items():
        p = PROFILES[name]
        xi = clip(p)
        ref = xi / float(1 << 23)
        frames = rs.tx(name, xi, scen=(40.0, 0, 0.0, 1.0), seed=1)
        res = Decoder(p).run(rs.to_q22(frames), len(xi), 0)       # 本家の受信
        s_gfm = snr(ref, res.pcm / float(1 << 23))
        stats = res.stats
        ok = stats["crc_fail"] == 0 and stats["lost"] == 0 and stats["high_crc_fail"] == 0 and abs(s_gfm - base) <= TOLERANCE_DB
        print(f"{'OK ' if ok else 'NG '} {name:11} 本家の受信 {s_gfm:6.2f} dB（基準 {base:.2f}）、{stats}")
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
