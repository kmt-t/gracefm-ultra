#!/usr/bin/env python3
"""
Rust のデコーダ (engine/gfm-rx) が、GraceFM の Python 実装 (gracefm.txrx.receive) と完全に一致することを確かめる。
送信 (Python の Transmitter) → 通信路 (雑音・フェージング・フレーム欠落) → 受信を、Python と Rust で行い、出力の PCM と品質コードを比べる。
受信は整数演算だけなので、完全一致するはず。

  .venv/bin/python tools/test_rs_rx_parity.py [--seconds 1.0] [--only B200+SUB100]
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
from gracefm_ultra.rx import GRACEFM  # noqa: E402,F401
from gracefm import channel as chan, modem  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm.txrx import Transmitter, receive  # noqa: E402
from gracefm_ultra.channels import loss_mask  # noqa: E402
from prep_haarnet import clip_pcm  # noqa: E402

BIN = os.path.join(HERE, "..", "engine", "target", "release", "gfm-rx")
SRC = os.path.join(GRACEFM, "eval_audio", "train_lossless")

# (主系, 副系, 主系の Es/N0, 副系の Es/N0, フェージング, 欠落率, 副系のずれ [サンプル])
CASES = [
    ("B200M", None, 9, None, "awgn", 0.0, 0),
    ("B200M", None, 5, None, "rayleigh", 0.1, 0),
    ("B200", None, 30, None, "awgn", 0.0, 0),
    ("B200", "SUB100", 8, 4, "rayleigh", 0.1, 0),
    ("B200", "SUB100", 6, 2, "rayleigh", 0.2, 10),
    ("M700", "SUB100-48K", 7, 3, "rayleigh", 0.1, 0),
    ("M700M", "SUB100-48K", 6, 4, "awgn", 0.05, 0),
    ("SUB100", None, 5, None, "rayleigh", 0.05, 0),
    ("SUB50M", None, 8, None, "awgn", 0.0, 0),
    # 基本 CRC が通らない境界付近の C/N（Chase 救済が多く効く）と、高域 CRC だけ落ちる領域
    ("B200M", None, 3, None, "awgn", 0.0, 0),
    ("B200", None, 2, None, "awgn", 0.0, 0),
    ("SUB100", None, 2, None, "awgn", 0.0, 0),
    ("M700", None, 3, None, "awgn", 0.0, 0),
    ("B200", "SUB100", 3, 1, "awgn", 0.0, 0),
]


def write_sym(path, frames):
    nf = len(frames)
    ns = next(len(f) for f in frames if f is not None)
    with open(path, "wb") as fh:
        fh.write(np.array([nf, ns], "<u4").tobytes())
        fh.write(bytes(0 if f is None else 1 for f in frames))
        for f in frames:
            if f is not None:
                fh.write(np.asarray(f, "<i4").tobytes())


def run_case(case, x, seed):
    main, sub, e_a, e_b, kind, loss, off = case
    pm, ps = PROFILES[main], (PROFILES[sub] if sub else None)
    rng = np.random.default_rng(seed)
    fa = Transmitter(pm, 0, 0).run(x)
    fa = chan.through(fa, e_a, chan.fading_gains(len(fa), kind, np.random.default_rng(seed + 1)), rng)
    mk = loss_mask(len(fa), loss, 4, np.random.default_rng(seed + 2))
    fa = [None if m else f for f, m in zip(fa, mk)]
    fb = None
    if ps:
        fb = Transmitter(ps, 1, off).run(x)
        fb = chan.through(fb, e_b, chan.fading_gains(len(fb), kind, np.random.default_rng(seed + 3)), rng)
        mk = loss_mask(len(fb), loss, 4, np.random.default_rng(seed + 4))
        fb = [None if m else f for f, m in zip(fb, mk)]
    y, st = receive(pm, fa, fb, len(x), off, profile_b=ps)
    with tempfile.TemporaryDirectory() as d:
        a, b, o, q = (os.path.join(d, n) for n in ("a.sym", "b.sym", "o.raw", "q.txt"))
        write_sym(a, fa)
        cmd = [BIN, "--profile", main, "--frames-a", a, "--n", str(len(x)), "--offset-b", str(off), "--out", o, "--quality", q]
        if ps:
            write_sym(b, fb)
            cmd += ["--profile-b", sub, "--frames-b", b]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return f"実行失敗: {r.stderr[-300:]}", False
        got = np.fromfile(o, "<i4").reshape(-1, 2).astype(np.int64)
        gq = np.array([int(v) for v in open(q).read().split()], np.int64)
    sa = st["A"]
    info = f"lost {sa['lost']} crc_fail {sa['crc_fail']} high_crc_fail {sa['high_crc_fail']} plc {st['plc_samples']} | rust: {r.stderr.strip().splitlines()[-1][:80]}"
    ok = got.shape == y.shape and np.array_equal(got, y) and np.array_equal(gq, np.asarray(st["quality"]))
    if not ok:
        n = int((got != y).any(axis=1).sum()) if got.shape == y.shape else -1
        info += f" 不一致: PCM {n} サンプル, 品質 {int((gq != np.asarray(st['quality'])).sum()) if len(gq) == len(st['quality']) else -1} ブロック"
    return info, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(SRC) if f.endswith((".flac", ".wav")))
    bad = 0
    for i, case in enumerate(CASES):
        name = case[0] + ("+" + case[1] if case[1] else "")
        if a.only and a.only != name:
            continue
        p = PROFILES[case[0]]
        nfr = int(a.seconds * p.fs) // p.block
        x = np.asarray(clip_pcm(os.path.join(SRC, files[3 + i]), p, nfr, 40000), np.int64)[:nfr * p.block]
        info, ok = run_case(case, x, 100 + i)
        print(f"{'OK ' if ok else 'NG '} {name:18} {case[2]}dB {case[4]:8} loss {case[5]} off {case[6]}: {info}", flush=True)
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
