#!/usr/bin/env python3
"""
HaarNet-lite 用の特徴量データを作る。data/<profile>/<split>/*.npz (gen_dataset の出力) から、全成分 (M, S)・全帯域の
仮数bit だけを取り出し、同じ音源を符号化し直して刻み (指数・アッテネータ込み) を求める。
係数の並びは Layout の coef_offset 順 (M の帯域 → S の帯域、bits>0 の帯域のみ)。
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



def clip_pcm(path, p, nfr, start):
    """gen_dataset.make_clip と同じ手順で、符号化に使った PCM24 を再現する"""
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


def dataset_index(p, lay):
    """coef_offset 順の係数番号 → gen_dataset の coef 配列 (F, 2, block) の平坦な添字"""
    starts = np.concatenate([[0], np.cumsum([p.ncoef(k) for k in range(p.nbands)])])
    idx = np.zeros(lay.n_coef, np.int64)
    group = np.zeros(lay.n_coef, np.int64)
    for gi, ((ch, k), o) in enumerate(lay.coef_offset.items()):
        n = p.ncoef(k)
        ci = 0 if ch == "M" else 1
        idx[o:o + n] = ci * p.block + starts[k] + np.arange(n)
        group[o:o + n] = gi
    return idx, group


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
    idx = lay.coef_stream_idx                                             # 全仮数bit の送信順の位置
    pos = lay.coef_pos
    coef = lay.coef_flat                                                   # bit → coef_offset 順の係数番号
    strong = (idx < lay.n_strong).astype(np.uint8)
    prot = np.zeros(len(lay.coef_items), bool)
    prot[lay.top_coef_idx] = True
    cidx, group = dataset_index(p, lay)
    outdir = os.path.join(a.data, a.profile, "hn", a.split)
    os.makedirs(outdir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.data, a.profile, a.split, "*.npz"))):
        d = np.load(f)
        nfr = d["bits"].shape[0]
        x = clip_pcm(os.path.join(a.src, str(d["source"])), p, nfr, int(d["start"]))
        step = steps_of(p, lay, x, nfr)
        tq = d["coef"].reshape(nfr, -1)[:, cidx].astype(np.float64)
        m = tq / step
        assert np.abs(m - np.rint(m)).max() < 1e-6, "刻みが一致しない (再符号化のずれ)"
        np.savez(os.path.join(outdir, os.path.basename(f)), llr=d["llr"][:, :, idx], bits=d["bits"][:, idx],
                 step=step, coef0=d["coef0"].reshape(nfr, -1)[:, cidx].astype(np.float32), coefq=tq.astype(np.float32),
                 lost=d["lost"], esn0=d["esn0"], var=d["var"], scen=d["scen"],
                 pos=pos, coef_of_bit=coef, strong=strong, protected=prot.astype(np.uint8), group=group)
        print(os.path.basename(f), nfr, "frames ok", flush=True)


if __name__ == "__main__":
    main()
