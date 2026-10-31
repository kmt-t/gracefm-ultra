#!/usr/bin/env python3
"""
欠落フレームの補間の評価 (時間領域、M 成分、32 kHz)。欠落した区間だけの SNR を、バースト長ごとに出す。
欠落以外は正解を使う (ビット誤りの影響とは切り離して、補間そのものを測る)。

  zero   欠落を 0 にする
  gfm    現行 GraceFM の LPC 外挿 (過去のみ、10 ms 保持 + 20 ms で減衰)
  ar2    AR モデルによる両側補間 (前後 ctx サンプルの文脈)
  net    学習した PLC ネットワーク (--ckpt を指定したとき)

  .venv/bin/python tools/eval_plc.py [--gaps 1,3,8,25] [--ckpt runs/baseline/plc.pt]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm import plc  # noqa: E402
from gracefm_ultra.plc_ar import interpolate  # noqa: E402
from tools.common import wavio  # noqa: E402

FS, BLOCK = 32000, 64


def val_clips(src, seconds, nfiles):
    files = sorted(f for f in os.listdir(src) if f.endswith((".flac", ".wav")))
    files = [f for i, f in enumerate(files) if i % 7 == 3][:nfiles]
    out = []
    for f in files:
        x, fs = wavio.read(os.path.join(src, f))
        x = wavio.resample(x, fs, FS)
        st = len(x) // 3
        m = (x[st:st + int(seconds * FS)] * 0.5).mean(1)
        out.append(wavio.to_pcm24(m[:, None])[:, 0].astype(float))
    return out


def gap_positions(n, g, rng, spacing=6144, ctx=2048):
    """長さ g フレームの欠落を、互いに spacing サンプル以上離して並べる"""
    pos, p = [], ctx
    while p + g * BLOCK + ctx < n:
        pos.append(p)
        p += g * BLOCK + spacing
    return pos


def methods(ckpt):
    m = {
        "zero": lambda x, a, b: np.zeros(b - a),
        "gfm": lambda x, a, b: _gfm(x, a, b),
        "ar2": lambda x, a, b: interpolate(x, a, b),
    }
    if ckpt:
        from gracefm_ultra.plcnet import load_predictor
        m["net"] = load_predictor(ckpt)
    return m


def _gfm(x, a, b):
    y = x.copy().astype(np.int64)
    valid = np.ones(len(x), bool)
    valid[a:b] = False
    lo, hi = max(0, a - 2048), min(len(x), b + 2048)
    out = plc.conceal(y[lo:hi, None], valid[lo:hi], FS)[:, 0]
    return out[a - lo:b - lo].astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(GRACEFM, "eval_audio", "train_lossless"))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--gaps", default="1,3,8,25")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--noise-db", type=float, default=0.0, help="文脈に加える雑音 (信号比 [dB] 負の値。0 なら無し)")
    a = ap.parse_args()
    clips = val_clips(a.src, a.seconds, a.files)
    meths = methods(a.ckpt)
    rng = np.random.default_rng(0)
    print(f"{len(clips)} 音源 × {a.seconds}s。欠落区間の SNR [dB] (区間ごとの dB の平均 / 全区間の合算)")
    print(f"{'gap':>5}{'ms':>6}{'n':>5}  " + "  ".join(f"{k:>14}" for k in meths))
    for g in [int(v) for v in a.gaps.split(",")]:
        rows = {k: [] for k in meths}
        for x in clips:
            xs = x.copy()
            if a.noise_db:
                xs = x + rng.normal(0, np.sqrt(np.mean(x ** 2) * 10 ** (a.noise_db / 10)), len(x))
            for p0 in gap_positions(len(x), g, rng):
                b = p0 + g * BLOCK
                ref = x[p0:b]
                if np.mean(ref ** 2) < 1e-2 * 2 ** 20:                # 無音に近い区間は除く
                    continue
                for k, fn in meths.items():
                    est = fn(xs, p0, b)
                    rows[k].append((np.sum(ref ** 2), np.sum((ref - est) ** 2)))
        n = len(rows["zero"])
        cells = []
        for k in meths:
            r = np.array(rows[k])
            per = 10 * np.log10(r[:, 0] / np.maximum(r[:, 1], 1e-9))
            cells.append(f"{per.mean():6.2f} /{10 * np.log10(r[:, 0].sum() / r[:, 1].sum()):6.2f}")
        print(f"{g:5d}{g * BLOCK / FS * 1000:6.0f}{n:5d}  " + "  ".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
