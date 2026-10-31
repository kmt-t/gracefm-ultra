#!/usr/bin/env python3
"""
Ultra の受信（Rust、浮動小数点、engine/gfm-rx）を、本家 GraceFM の受信（Python、整数）と、送信した原音に対する SNR で比べる。
同じ受信シンボル（雑音・フェージング・フレーム欠落つき）を両方に通し、原音（出力の遅れを補償）との SNR を、全体と、欠落補間を含まない区間で出す。

  .venv/bin/python tools/eval_float_rx.py [--seconds 1.0]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.gfm import GRACEFM  # noqa: E402
from gracefm import channel as chan  # noqa: E402
from gracefm.encoder import Encoder  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm.receiver import receive  # noqa: E402
from gracefm_ultra.channels import loss_mask  # noqa: E402
from ref import clip_pcm  # noqa: E402

BIN = os.path.join(HERE, "..", "engine", "target", "release", "gfm-rx")
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


def write_sym(path, frames):
    """フレームファイル: u32 フレーム数, u32 シンボル数, フラグ, 届いたフレームの I/Q (i32)"""
    ns = next(len(f) for f in frames if f is not None)
    with open(path, "wb") as fh:
        fh.write(np.array([len(frames), ns], "<u4").tobytes())
        fh.write(bytes(0 if f is None else 1 for f in frames))
        for f in frames:
            if f is not None:
                fh.write(np.asarray(f, "<i4").tobytes())


def snr(ref, y, mask=None):
    e, s = (y - ref) ** 2, ref ** 2
    if mask is not None:
        e, s = e[mask], s[mask]
    return 10 * np.log10(s.sum() / max(e.sum(), 1e-30))


def run(case, x, seed):
    main, sub, e_a, e_b, kind, loss = case
    pm, ps = PROFILES[main], (PROFILES[sub] if sub else None)
    rng = np.random.default_rng(seed)
    fa = Encoder(pm, 0, 0).run(x)
    fa = chan.through(fa, e_a, chan.fading_gains(len(fa), kind, np.random.default_rng(seed + 1)), rng)
    mk = loss_mask(len(fa), loss, 4, np.random.default_rng(seed + 2))
    fa = [None if m else f for f, m in zip(fa, mk)]
    fb = None
    if ps:
        fb = Encoder(ps, 1, 0).run(x)
        fb = chan.through(fb, e_b, chan.fading_gains(len(fb), kind, np.random.default_rng(seed + 3)), rng)
        mk = loss_mask(len(fb), loss, 4, np.random.default_rng(seed + 4))
        fb = [None if m else f for f, m in zip(fb, mk)]
    y_gfm, st = receive(pm, fa, fb, len(x), 0, profile_b=ps)
    with tempfile.TemporaryDirectory() as d:
        a, b, o, q = (os.path.join(d, n) for n in ("a.sym", "b.sym", "o.raw", "q.txt"))
        write_sym(a, fa)
        cmd = [BIN, "--profile", main, "--frames-a", a, "--n", str(len(x)), "--out", o, "--quality", q]
        if ps:
            write_sym(b, fb)
            cmd += ["--profile-b", sub, "--frames-b", b]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-300:])
        y_ultra = np.fromfile(o, "<f4").reshape(-1, 2).astype(np.float64)
        q_ultra = np.array([int(v) for v in open(q).read().split()])
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
        x = np.asarray(clip_pcm(os.path.join(SRC, files[3 + i]), p, nfr, 40000), np.int64)[:nfr * p.block]
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
