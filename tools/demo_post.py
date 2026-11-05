#!/usr/bin/env python3
"""
PostNet の試聴用の音。data/<profile>/post/awgn_val の区間から、指定した Es/N0 ごとに、原音・復号のまま・MendNet・PostNet を
ステレオ 32 kHz の WAV (float32、正規化しない) で書く。PostNet は超解像をかける判定 (postnet.gate) がオフのところは MendNet のまま。

  .venv/bin/python tools/demo_post.py --ckpt runs/post_B200.pt --match Chopin_-_Waltz,Scarlatti,C418_-_Mixes --esn0 18,20,30
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm_ultra.postnet import Transform  # noqa: E402
from eval_post import load, lr, process, wave_of  # noqa: E402
from eval_mendnet import CN_OFFSET  # noqa: E402

FS = 32000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--match", required=True, help="区間のファイル名に含む文字列 (カンマ区切り)。それぞれ最初の 1 区間を使う")
    ap.add_argument("--esn0", default="18,20,30", help="Es/N0 [dB] (カンマ区切り)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "demo_post"))
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, tf, tfc = load(a.ckpt, dev), Transform().to(dev), Transform()
    d = os.path.join(a.data, a.profile)
    files = sorted(glob.glob(os.path.join(d, "post", "awgn_val", "*.npz")))
    os.makedirs(a.out, exist_ok=True)
    for key in a.match.split(","):
        f = next(x for x in files if key in os.path.basename(x))
        z = np.load(f)
        dec = np.load(os.path.join(d, "mend_awgn", "val", os.path.basename(f)))["dec"]
        name = key.strip("_-.")
        sf.write(os.path.join(a.out, f"{name}_原音.wav"), lr(wave_of(z["truth"], tfc)).astype(np.float32), FS, subtype="FLOAT")
        for e in (float(x) for x in a.esn0.split(",")):
            sc = int(np.flatnonzero(np.isclose(z["scen"][:, 0], e))[0])
            out, g = process(net, tf, z["e0"][sc], z["sd"][sc], z["status"][sc], dev)
            out["復号のまま"] = wave_of(dec[sc], tfc)
            cn = f"CN{e + CN_OFFSET:g}"
            for k in ("復号のまま", "MendNet", "PostNet"):
                sf.write(os.path.join(a.out, f"{name}_{cn}_{k}.wav"), lr(out[k]).astype(np.float32), FS, subtype="FLOAT")
            print(f"{name} {cn}: 判定がオン {g.mean():.0%}", flush=True)
    print(f"→ {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
