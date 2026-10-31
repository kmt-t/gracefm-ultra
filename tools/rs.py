"""Rust のコマンド（engine/gfm-tx: 送信と通信路、engine/gfm-rx: Ultra の受信）を Python から呼ぶ補助。"""
import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "engine", "target", "release")


def write_pcm(path, pcm24):
    """符号つき 24 bit 整数の (n, 2) を int32 の生データにする（gfm-tx / gfm-gen の入力）"""
    np.asarray(pcm24, "<i4").tofile(path)


def read_sym(path):
    """gfm-tx の出力 (.sym) → フレームのリスト（届かなければ None、届けば (シンボル数, 2) の実数）"""
    d = np.fromfile(path, np.uint8)
    nf, ns = np.frombuffer(d[:8].tobytes(), "<u4")
    flags = d[8:8 + nf]
    data = np.frombuffer(d[8 + nf:].tobytes(), "<f4").reshape(-1, ns, 2).astype(np.float64)
    out, k = [], 0
    for f in flags:
        out.append(data[k] if f else None)
        if f:
            k += 1
    return out


def to_q22(frames):
    """実数のシンボル (±1 = 1.0) → 本家 Python の受信が受け取る整数 (±1 = 2^22)"""
    return [None if f is None else np.rint(f * (1 << 22)).astype(np.int64) for f in frames]


def write_sym(path, frames):
    """フレームのリスト（実数）→ .sym"""
    ns = next(len(f) for f in frames if f is not None)
    with open(path, "wb") as fh:
        fh.write(np.array([len(frames), ns], "<u4").tobytes())
        fh.write(bytes(0 if f is None else 1 for f in frames))
        for f in frames:
            if f is not None:
                fh.write(np.asarray(f, "<f4").tobytes())


def tx(profile, pcm24, scen=(30.0, 0, 0.0, 1.0), seed=1, stream_id=0, offset=0, si=0):
    """送信と通信路。scen = (基準 Es/N0 [dB], Rayleigh か (0/1), 欠落率, 平均バースト長)。返り値: フレームのリスト（実数）"""
    with tempfile.TemporaryDirectory() as d:
        raw, out = os.path.join(d, "x.raw"), os.path.join(d, "a.sym")
        write_pcm(raw, pcm24)
        cmd = [os.path.join(BIN, "gfm-tx"), "--profile", profile, "--pcm", raw, "--seed", str(seed), "--stream-id", str(stream_id),
               "--offset", str(offset), "--si", str(si), "--scen", ",".join(str(float(v)) for v in scen), "--out", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-400:])
        return read_sym(out)


def rx(profile, frames_a, n, frames_b=None, profile_b=None, offset_b=0):
    """Ultra の受信（浮動小数点）。返り値: (音声 (n, 2) 満点 ±1、品質コード（出力ブロックごと）、標準エラー出力の統計行)"""
    with tempfile.TemporaryDirectory() as d:
        a, b, o, q = (os.path.join(d, v) for v in ("a.sym", "b.sym", "o.raw", "q.txt"))
        write_sym(a, frames_a)
        cmd = [os.path.join(BIN, "gfm-rx"), "--profile", profile, "--frames-a", a, "--n", str(n), "--offset-b", str(offset_b), "--out", o, "--quality", q]
        if frames_b is not None:
            write_sym(b, frames_b)
            cmd += ["--profile-b", profile_b, "--frames-b", b]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-400:])
        y = np.fromfile(o, "<f4").reshape(-1, 2).astype(np.float64)
        quality = np.array([int(v) for v in open(q).read().split()])
        return y, quality, r.stderr.strip()
