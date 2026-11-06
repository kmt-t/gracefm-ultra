"""Rust のコマンド（engine/gfm-tx: 送信と通信路）を Python から呼ぶ補助。"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.gfm import GRACEFM  # noqa: E402
from tools.common import wavio  # noqa: E402

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


def clip(p, seconds=2.0, index=3, start=10.0):
    """学習用音源（GraceFM の eval_audio/train_lossless）の index 番目の start 秒から seconds 秒を、プロファイル p のサンプリング周波数で、符号つき 24 bit 整数 (n, 2) にする"""
    src = os.path.join(GRACEFM, "eval_audio", "train_lossless")
    path = os.path.join(src, sorted(f for f in os.listdir(src) if f.endswith((".flac", ".wav")))[index])
    x, fs = sf.read(path, dtype="float64", always_2d=True, start=int(start * 44100), frames=int((seconds + 1.5) * 44100))
    xs = wavio.resample(x[:, :2], fs, p.fs)[int(0.5 * p.fs):int(0.5 * p.fs) + int(seconds * p.fs)] * 0.5
    xi = wavio.to_pcm24(xs)
    return xi[:len(xi) // p.block * p.block]
