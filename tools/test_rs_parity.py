#!/usr/bin/env python3
"""
Rust の生成器 (engine/gfm-gen) が、Python の GraceFM (符号化・フレーム化・変調・復調) と一致することを確かめる。
通信路の雑音は両方で Es/N0 = 300 dB (雑音なし) にして、決定的な部分を完全一致で比べる:
  送信ビット (全位置)、刻み、量子化前後の係数、受信 LLR (全位置)、雑音分散。
乱数を使う部分 (雑音、フェージング、欠落) は一致しないので、別に統計で確かめる (--stats)。

  .venv/bin/python tools/test_rs_parity.py [--profiles B200,SUB100] [--seconds 1.5]
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
from gracefm import channel as chan, framing, modem  # noqa: E402
from gracefm.codec import encode_block, haar_forward  # noqa: E402
from gracefm.decoder import reconstruct_coeffs  # noqa: E402
from gracefm.layout import Layout  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from prep_haarnet import clip_pcm, dataset_index, steps_of  # noqa: E402

BIN = os.path.join(HERE, "..", "engine", "target", "release", "gfm-gen")
SRC = os.path.join(GRACEFM if isinstance(GRACEFM, str) else "../gracefm", "eval_audio", "train_lossless")


def python_ref(p, x, nfr, e0):
    lay = Layout(p)
    bps = modem.bits_per_symbol(p.modulation)
    nsym = modem.n_symbols(lay.n_tx, lay.n_strong, p.modulation)
    scr = modem.prbs(lay.n_tx)
    rng = np.random.default_rng(1)
    bits = np.zeros((nfr, lay.n_tx), np.uint8)
    llr = np.zeros((nfr, lay.n_tx), np.int8)
    var = np.zeros(nfr, np.int64)
    coef0 = np.zeros((nfr, 2, p.block), np.int64)
    coefq = np.zeros((nfr, 2, p.block), np.int64)
    for i in range(nfr):
        blk = x[i * p.block:(i + 1) * p.block]
        bd = encode_block(blk, p)
        bt = framing.pack(lay, bd, 0, i, None)
        bits[i] = bt
        buf = np.zeros(nsym * bps, np.int64)
        buf[:len(bt)] = bt
        buf ^= modem.prbs(len(buf))
        sym = modem.modulate(buf, nsym, p.modulation, p.alpha)
        rx = chan.adc(chan.awgn(sym / float(1 << modem.SYM_FRAC), e0, rng))
        l, _, v = modem.demodulate(rx[modem.N_PREAMBLE:], lay.n_tx, p.modulation, p.alpha)
        llr[i] = np.clip(np.where(scr == 1, -l, l), -127, 127)
        var[i] = v
        co = reconstruct_coeffs(bd, p)
        ms = {"M": (blk[:, 0] + blk[:, 1] + 2) >> 2, "S": (blk[:, 0] - blk[:, 1] + 2) >> 2}
        for ci, ch in enumerate(("M", "S")):
            coefq[i, ci] = np.concatenate(co[ch])
            coef0[i, ci] = np.concatenate(haar_forward(ms[ch], p.levels))
    cidx, group = dataset_index(p, lay)
    return dict(bits_full=bits, llr_full=llr, var=var, coef0=coef0.reshape(nfr, -1)[:, cidx], coefq=coefq.reshape(nfr, -1)[:, cidx],
                step=steps_of(p, lay, x, nfr), group=group, pos=lay.coef_pos, coef_of_bit=lay.coef_flat,
                strong=(lay.coef_stream_idx < lay.n_strong).astype(np.uint8),
                protected=np.isin(np.arange(len(lay.coef_items)), lay.top_coef_idx).astype(np.uint8),
                bits=bits[:, lay.coef_stream_idx], llr=llr[:, lay.coef_stream_idx])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", default="B200M,B200,M700M,M700,SUB100,SUB50M,SUB100-48K")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--e0", type=float, default=300.0)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(SRC) if f.endswith((".flac", ".wav")))
    bad = 0
    for name in a.profiles.split(","):
        p = PROFILES[name]
        nfr = int(a.seconds * p.fs) // p.block
        x = clip_pcm(os.path.join(SRC, files[3]), p, nfr, 40000)
        x = np.asarray(x, np.int64)[:nfr * p.block]
        ref = python_ref(p, x, nfr, a.e0)
        with tempfile.TemporaryDirectory() as d:
            raw, out = os.path.join(d, "x.raw"), os.path.join(d, "o.npz")
            x.astype("<i4").tofile(raw)
            r = subprocess.run([BIN, "--profile", name, "--pcm", raw, "--scen", f"{a.e0},0,0,1", "--seed", "1", "--out", out, "--full"],
                               capture_output=True, text=True)
            if r.returncode:
                print(name, "実行失敗:", r.stderr[-300:])
                bad += 1
                continue
            d_ = np.load(out)
            res = []
            for k, v in ref.items():
                got = d_["llr_full"][0] if k == "llr_full" else d_["llr"][0] if k == "llr" else d_["var"][0] if k == "var" else d_[k]
                got = np.asarray(got).reshape(np.shape(v))
                n = int(np.sum(got.astype(np.float64) != np.asarray(v, np.float64)))
                res.append((k, n))
            fails = [(k, n) for k, n in res if n]
            print(f"{name:12} {nfr} frames, n_tx {ref['bits_full'].shape[1]}: " + ("全て一致" if not fails else "不一致 " + ", ".join(f"{k}:{n}" for k, n in fails)))
            bad += bool(fails)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
