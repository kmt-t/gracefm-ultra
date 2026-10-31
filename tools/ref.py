"""試験・デモ用の参照: 本家 GraceFM（Python）で、音源の区間の PCM と量子化の刻みを再現する。Rust の送信（gfm-gen）との一致確認に使う。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.gfm import GRACEFM  # noqa: E402,F401
from gracefm.codec import FRAC, attenuator_expanded, encode_block, exp_range  # noqa: E402
from tools.common import wavio  # noqa: E402


def clip_pcm(path, p, nfr, start):
    """試験用: 音源の先頭付近（最初の seconds + 2 秒）を再標本化して、start サンプル目から nfr フレームぶんの PCM24 を作る"""
    x, fs = wavio.read(path)
    seconds = nfr * p.block / p.fs
    n = int(seconds * p.fs)
    xs = wavio.resample(x[: int((seconds + 2) * fs * 1.0) + 1 if len(x) > 3 * fs * seconds else len(x)], fs, p.fs)
    return wavio.to_pcm24(xs[start:start + n] * 0.5)


def steps_of(p, lay, x, nfr):
    """全係数 (lay.n_coef 個、coef_offset 順) の刻み"""
    out = np.ones((nfr, lay.n_coef), np.float32)
    b = p.block
    for i in range(nfr):
        bd = encode_block(x[i * b:(i + 1) * b], p)
        for (ch, k), o in lay.coef_offset.items():
            bits = {"M": p.bits_m, "S": p.bits_s}[ch][k]
            n = p.ncoef(k)
            a = attenuator_expanded(bd.attn[ch][k], p.attenuator_group(ch, k), n)
            out[i, o:o + n] = 2.0 ** (int(bd.eidx[ch][k]) + exp_range(k, bits, p)[0] + FRAC - np.asarray(a, float))
    return out
