#!/usr/bin/env python3
"""
Ultra の受信（Rust、浮動小数点、engine/gfm-rx）を、本家 GraceFM の受信（Python、整数）と、送信した原音に対する SNR で比べる。
同じ受信シンボル（雑音・フェージング・フレーム欠落つき）を両方に通し、原音（出力の遅れを補償）との SNR を、全体と、欠落補間を含まない区間で出す。

  .venv/bin/python tools/eval_float_rx.py [--seconds 1.0]
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.gfm import GRACEFM  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm.receiver import receive  # noqa: E402
from tools.common import wavio  # noqa: E402
import rs  # noqa: E402

SRC = os.path.join(GRACEFM, "eval_audio", "train_lossless")
DELAY = {32000: 3, 48000: 5}
# (主系, 副系, 主系の Es/N0, 副系の Es/N0, フェージング, 欠落率)
CASES = [
    ("B200", None, 30, None, "awgn", 0.0),
    ("B200", None, 12, None, "awgn", 0.0),
    ("B200", None, 8, None, "awgn", 0.0),
    ("B200", None, 6, None, "rayleigh", 0.1),
    ("B200", "SUB100", 8, 4, "rayleigh", 0.1),
    ("B200", "SUB100", 5, 2, "rayleigh", 0.2),
    ("M700", "SUB100-48K", 7, 3, "rayleigh", 0.1),
    ("SUB100", None, 6, None, "awgn", 0.0),
]


def clip(path, p, nfr):
    """音源の 10 秒目から nfr フレームぶん（符号つき 24 bit 整数 (n, 2)）"""
    n = nfr * p.block
    x, fs = sf.read(path, dtype="float64", always_2d=True, start=int(10 * 44100), frames=int(n / p.fs * 44100) + 2 * 44100)
    return wavio.to_pcm24(wavio.resample(x[:, :2], fs, p.fs)[int(0.5 * p.fs):int(0.5 * p.fs) + n] * 0.5)


def snr(ref, y, mask=None):
    e, s = (y - ref) ** 2, ref ** 2
    if mask is not None:
        e, s = e[mask], s[mask]
    return 10 * np.log10(s.sum() / max(e.sum(), 1e-30))


def run(case, x, seed):
    """同じ受信シンボルを、本家の受信（Python、整数）と Ultra の受信（Rust、浮動小数点）にかける。返り値: {名前: (音声（満点 ±1）, 品質コード)}"""
    main, sub, e_a, e_b, kind, loss = case
    pm, ps = PROFILES[main], (PROFILES[sub] if sub else None)
    fade = int(kind == "rayleigh")
    fa = rs.tx(main, x, scen=(e_a, fade, loss, 4), seed=seed, stream_id=0, si=0)
    fb = rs.tx(sub, x, scen=(e_b, fade, loss, 4), seed=seed, stream_id=1, si=1) if ps else None
    y_gfm, st = receive(pm, rs.to_q22(fa), rs.to_q22(fb) if ps else None, len(x), 0, profile_b=ps)
    y_ultra, q_ultra, _ = rs.rx(main, fa, len(x), fb, sub)
    return {"gfm": (y_gfm / float(1 << 23), np.asarray(st["quality"])), "ultra": (y_ultra, q_ultra)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=1.0)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(SRC) if f.endswith((".flac", ".wav")))
    print(f"{'ケース':34}{'全体 SNR [dB] 本家 → Ultra':>28}{'品質コード 0 のブロックだけ':>30}")
    for i, case in enumerate(CASES):
        p = PROFILES[case[0]]
        nfr = int(a.seconds * p.fs) // p.block
        x = clip(os.path.join(SRC, files[3 + i]), p, nfr)
        ref = x / float(1 << 23)
        res = run(case, x, 200 + i)
        d = DELAY[p.fs]
        rdel = np.concatenate([np.zeros((d, 2)), ref[:-d]])        # 出力は d サンプル遅れる
        out = {}
        for mode, (y, q) in res.items():
            ok = np.repeat(q == 0, p.block)[:len(ref)]
            out[mode] = (snr(rdel, y), snr(rdel, y, ok) if ok.any() else float("nan"))
        name = case[0] + ("+" + case[1] if case[1] else "") + f" {case[2]}dB {case[4]} loss{case[5]}"
        print(f"{name:34}{out['gfm'][0]:12.2f} → {out['ultra'][0]:6.2f}{out['gfm'][1]:14.2f} → {out['ultra'][1]:6.2f}", flush=True)


if __name__ == "__main__":
    main()
