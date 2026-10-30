#!/usr/bin/env python3
"""
HaarNet-lite 用の特徴量データを作る。data/<profile>/<split>/*.npz (gen_dataset の出力) から、M 成分の k=0..2 の
仮数bit (140個) だけを取り出し、同じ音源を符号化し直して刻み (指数・アッテネータ込み) を求める。
指数・ヘッダは復号済みとして正解を使う (試作の仮定)。出力: data/<profile>/hn/<split>/<name>.npz

  .venv/bin/python tools/prep_haarnet.py --split train
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm.codec import FRAC, attenuator_expanded, encode_block, exp_range  # noqa: E402
from gracefm.layout import Layout  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402

NB = 3                                   # k = 0..2


def clip_pcm(path, p, nfr, start):
    """gen_dataset.make_clip と同じ手順で、符号化に使った PCM24 を再現する"""
    x, fs = wavio.read(path)
    seconds = nfr * p.block / p.fs
    n = int(seconds * p.fs)
    xs = wavio.resample(x[: int((seconds + 2) * fs * 1.0) + 1 if len(x) > 3 * fs * seconds else len(x)], fs, p.fs)
    return wavio.to_pcm24(xs[start:start + n] * 0.5)


def steps_of(p, x, nfr):
    out = np.ones((nfr, sum(p.ncoef(k) for k in range(NB))), np.float32)
    b = p.block
    for i in range(nfr):
        bd = encode_block(x[i * b:(i + 1) * b], p)
        o = 0
        for k in range(NB):
            n = p.ncoef(k)
            a = attenuator_expanded(bd.attn["M"][k], p.attenuator_group("M", k), n)
            out[i, o:o + n] = 2.0 ** (int(bd.eidx["M"][k]) + exp_range(k, p.bits_m[k], p)[0] + FRAC - np.asarray(a, float))
            o += n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M")
    ap.add_argument("--split", default="val")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--src", default=os.path.join(GRACEFM, "eval_audio", "train_lossless"))
    a = ap.parse_args()
    if len(os.sched_getaffinity(0)) > 6:
        os.sched_setaffinity(0, {0, 1, 5, 6, 7, 11})
    p = PROFILES[a.profile]
    lay = Layout(p)
    sel = np.array([i for i, t in enumerate(lay.coef_items) if t[0] == "M" and t[1] < NB])
    idx = lay.coef_stream_idx[sel]
    pos = lay.coef_pos[sel]
    coef = np.array([lay.coef_flat[i] for i in sel])                      # 0..15 (M の k=0..2 が先頭 16 個)
    strong = (idx < lay.n_strong).astype(np.uint8)
    prot = np.zeros(len(lay.coef_items), bool)
    prot[lay.top_coef_idx] = True
    outdir = os.path.join(a.data, a.profile, "hn", a.split)
    os.makedirs(outdir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.data, a.profile, a.split, "*.npz"))):
        d = np.load(f)
        nfr = d["bits"].shape[0]
        x = clip_pcm(os.path.join(a.src, str(d["source"])), p, nfr, int(d["start"]))
        step = steps_of(p, x, nfr)
        tq = d["coef"][:, 0, :16].astype(np.float64)
        m = tq / step
        assert np.abs(m - np.rint(m)).max() < 1e-6, "刻みが一致しない (再符号化のずれ)"
        np.savez(os.path.join(outdir, os.path.basename(f)), llr=d["llr"][:, :, idx], bits=d["bits"][:, idx],
                 step=step, coef0=d["coef0"][:, 0, :16].astype(np.float32), coefq=tq.astype(np.float32),
                 lost=d["lost"], esn0=d["esn0"], var=d["var"], scen=d["scen"],
                 pos=pos, coef_of_bit=coef, strong=strong, protected=prot[sel].astype(np.uint8))
        print(os.path.basename(f), nfr, "frames ok", flush=True)


if __name__ == "__main__":
    main()
