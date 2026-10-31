#!/usr/bin/env python3
"""
現行 GraceFM 受信機 (../gracefm の receive) のベースライン測定。
主系 1系統を、Es/N0 の掃引・ランダム欠落・バースト欠落の通信路に通し、SNR と セグメンタルSNR を出す。

  .venv/bin/python tools/baseline.py [--profile B200M] [--seconds 8] [--seed 1]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))

from gracefm import channel as chan, modem  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm.txrx import Transmitter, receive  # noqa: E402
from tools.common import metrics, wavio  # noqa: E402
from gracefm_ultra.channels import loss_mask  # noqa: E402

AUDIO = ["lossless/Emotional_piano.wav", "lossless/Ceramic_Djembe.wav",
         "lossless/344211_giomilko_c-major-9-bossa-nova-guitar.wav"]


def run(x, p, esn0, rate, burst, seed):
    rng = np.random.default_rng(seed)
    xi = wavio.to_pcm24(x)
    fr = Transmitter(p, 0, 0).run(xi)
    mask = loss_mask(len(fr), rate, burst, rng)
    rx = [None if m else chan.adc(chan.awgn(f / float(1 << modem.SYM_FRAC), esn0, rng)) for f, m in zip(fr, mask)]
    y, st = receive(p, rx, None, len(x), 0)
    return wavio.from_pcm24(y), st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M", choices=PROFILES)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--esn0", default="0,3,6,9,12,20", help="AWGN 掃引の Es/N0 [dB] (カンマ区切り)")
    ap.add_argument("--rates", default="0.02,0.10,0.30", help="欠落率 (カンマ区切り)")
    ap.add_argument("--bursts", default="1,5,25", help="平均バースト長 (カンマ区切り)")
    a = ap.parse_args()
    p = PROFILES[a.profile]

    clips = []
    for f in AUDIO:
        x, fs = wavio.read(os.path.join(GRACEFM, "eval_audio", f))
        x = wavio.resample(x, fs, p.fs)[:int(a.seconds * p.fs)]
        clips.append(x * 0.5)

    def evaluate(esn0, rate, burst):
        s, g = [], []
        for i, x in enumerate(clips):
            y, _ = run(x, p, esn0, rate, burst, a.seed + i)
            ref = wavio.from_pcm24(wavio.to_pcm24(x))
            n = min(len(ref), len(y))
            s.append(metrics.snr_db(ref[:n], y[:n]))
            g.append(metrics.seg_snr_db(ref[:n, 0], y[:n, 0], p.fs))
        return np.mean(s), np.mean(g)

    print(f"profile={a.profile} fs={p.fs} clips={len(clips)} x {a.seconds}s  (SNR / segSNR [dB], 3素材平均)")
    print("\n[A] AWGN 掃引 (欠落なし)")
    for e in [float(v) for v in a.esn0.split(",") if v]:
        print(f"  Es/N0={e:>3} dB : SNR {evaluate(e, 0, 1)[0]:6.2f}  segSNR {evaluate(e, 0, 1)[1]:6.2f}" if False else
              "  Es/N0={:>3} dB : SNR {:6.2f}  segSNR {:6.2f}".format(e, *evaluate(e, 0, 1)))
    print("\n[B] フレーム欠落 (Es/N0=20 dB)  rate × 平均バースト長[フレーム]")
    for rate in [float(v) for v in a.rates.split(",") if v]:
        for burst in [float(v) for v in a.bursts.split(",") if v]:
            print("  rate={:.2f} burst={:>2} : SNR {:6.2f}  segSNR {:6.2f}".format(rate, burst, *evaluate(20, rate, burst)))


if __name__ == "__main__":
    main()
