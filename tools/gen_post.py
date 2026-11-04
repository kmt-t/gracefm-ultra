#!/usr/bin/env python3
"""
PostNet の学習・評価データ。data/<profile>/<src>/<split> (gen_hn.py --mend) の各区間・各シナリオを、確定した MendNet で丸ごと復元し、
data/<profile>/post/<out>/ に書く。窓 WIN フレームを WIN/2 ずつずらし、各窓の中央 WIN/2 フレームを採る (先頭と末尾の窓は端まで)。

出力の .npz (シナリオ S、フレーム T、係数 128 = M 64 + S 64、満点 1.0):
- `e0` (S,T,128) f32: MendNet の係数 (事後平均)
- `sd` (S,T,128) f16: その事後標準偏差 (量子化の雑音を含む)
- `truth` (T,128) f32: 原音の係数、`status` (S,T) u8: フレームの状態、`scen` (S,4) f32

  .venv/bin/python tools/gen_post.py --ckpt runs/mendbits_B200_v9.pt --src mend --split train --out train
  .venv/bin/python tools/gen_post.py --ckpt runs/mendbits_B200_v9.pt --src mend --split val --out val
  .venv/bin/python tools/gen_post.py --ckpt runs/mendbits_B200_v9.pt --src mend_awgn --split val --out awgn_val
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.haarnet import make_meta, soft_e0  # noqa: E402
from gracefm_ultra.mendnet import Q22, WIN, run_both  # noqa: E402
from eval_mendnet import load  # noqa: E402
from mend_windows import Windows  # noqa: E402

HOP = WIN // 2


def restore(win, sc, net, meta, chunk=64):
    """区間 0 のシナリオ sc を丸ごと復元する → (e0 (T,128), sd (T,128))"""
    n = win.truth[0].shape[0]
    starts = list(range(0, n - WIN + 1, HOP))
    if starts[-1] != n - WIN:
        starts.append(n - WIN)
    e0, sd = np.zeros((n, 128), np.float32), np.zeros((n, 128), np.float32)
    for i in range(0, len(starts), chunk):
        locs = [(0, sc, a) for a in starts[i:i + chunk]]
        per = [win.gather(locs, [ch] * len(locs)) for ch in range(2)]
        with torch.no_grad():
            llr, step = run_both(net, per, win.bit_idx.to(win.device), win.bit_ok.to(win.device), meta.nbit)
            m, s = soft_e0(llr, step, meta)
        m, s = (m / Q22).cpu().numpy(), (s / Q22).cpu().numpy()
        for j, (_, _, a) in enumerate(locs):
            lo = 0 if a == 0 else WIN // 4
            hi = WIN if a == n - WIN else WIN // 4 + HOP
            e0[a + lo:a + hi], sd[a + lo:a + hi] = m[j, lo:hi], s[j, lo:hi]
    return e0, sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True, help="確定した MendNet のチェックポイント")
    ap.add_argument("--src", default="mend")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help="出力先 data/<profile>/post/<out>")
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(a.data, a.profile, a.src, a.split, "*.npz")))
    outdir = os.path.join(a.data, a.profile, "post", a.out)
    os.makedirs(outdir, exist_ok=True)
    z = np.load(files[0])
    meta = make_meta({k: z[k] for k in ("pos", "coef_of_bit", "group")}, PROFILES[a.profile]).to(dev)
    net = load(a.ckpt, dev)
    t0 = time.time()
    for k, f in enumerate(files, 1):
        out = os.path.join(outdir, os.path.basename(f))
        if os.path.exists(out):
            continue
        win = Windows([f], dev)
        S = win.status[0].shape[0]
        res = [restore(win, sc, net, meta) for sc in range(S)]
        z = np.load(f)
        np.savez(out + ".tmp.npz", e0=np.stack([r[0] for r in res]), sd=np.stack([r[1] for r in res]).astype(np.float16),
                 truth=z["truth"], status=z["status"], scen=z["scen"])
        os.replace(out + ".tmp.npz", out)
        print(f"[{k}/{len(files)}] {os.path.basename(f)} (経過 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
