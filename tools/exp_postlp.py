#!/usr/bin/env python3
"""
現行 GraceFM 受信機の出力 (ビット誤りあり) に、Haar 合成のあとのローパスをかけると、聴感指標 (ViSQOL) と SNR がどうなるか。
Haar の漏れ (低域の係数の誤差が広帯域の雑音になる) への簡易な対策の効果を見る。ピアノ、B200M、AWGN。

  .venv/bin/python tools/exp_postlp.py --esn0 8,12,20
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from baseline import GRACEFM, run  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M")
    ap.add_argument("--esn0", default="8,12,20")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--cutoffs", default="12000,8000,6000,4000")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "demo"))
    a = ap.parse_args()
    from visqol import VisqolApi
    api = VisqolApi()
    api.create(mode="audio")
    p = PROFILES[a.profile]
    x, fs = wavio.read(os.path.join(GRACEFM, "eval_audio", "lossless", "Emotional_piano.wav"))
    x = wavio.resample(x, fs, p.fs)[int(2 * p.fs):int((2 + a.seconds) * p.fs)] * 0.5
    ref = wavio.from_pcm24(wavio.to_pcm24(x)).mean(axis=1)                  # B200M は M 成分 (左右の平均) だけを送る
    ref48 = wavio.resample(ref, p.fs, 48000)
    os.makedirs(a.out, exist_ok=True)
    cuts = [int(v) for v in a.cutoffs.split(",")]
    print(f"{a.profile} ピアノ {a.seconds}s。SNR [dB] / ViSQOL (現行受信機の出力に、ゼロ位相4次ローパスを後段でかける)\n")
    print(f"{'Es/N0':>6}{'処理なし':>16}" + "".join(f"{('LP ' + str(c // 1000) + 'k'):>16}" for c in cuts))
    for e in [float(v) for v in a.esn0.split(",")]:
        y, st = run(x, p, e, 0, 1, 1)
        y = y.mean(axis=1)[:len(ref)]
        cells = []
        for fc in [None] + cuts:
            z = y if fc is None else sosfiltfilt(butter(4, fc / (p.fs / 2), output="sos"), y)
            snr = 10 * np.log10(np.sum(ref[:len(z)] ** 2) / max(np.sum((ref[:len(z)] - z) ** 2), 1e-12))
            mos = float(api.measure_from_arrays(ref48, wavio.resample(z, p.fs, 48000), sample_rate=48000).moslqo)
            cells.append(f"{snr:6.2f} /{mos:6.3f}")
            if fc in (None, 6000):
                wavio.write_pcm24(os.path.join(a.out, f"rx_{a.profile}_es{int(e)}_{'raw' if fc is None else 'lp6k'}.wav"), np.clip(z, -1, 1), p.fs)
        print(f"{e:6.0f}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
