#!/usr/bin/env python3
"""
MendNet の聴き比べ用 WAV。検証データ (gen_hn.py --mend) の 1 シナリオについて、原音・受信のまま (使えないフレームは無音)・AR 補間・MendNet を
ステレオ (32 kHz、f32) で書く。MendNet は窓 100 フレームを 50 フレームずつずらし、各窓の中央 50 フレームを採る。

  .venv/bin/python tools/demo_mend.py --ckpt runs/mend_B200.pt --scen 2 --seconds 12 --out runs/demo_mend
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from eval_mend import ar_fill  # noqa: E402
from gracefm_ultra.mendnet import WIN, MendNet, build_input, compose, haar_inv, join_ar  # noqa: E402
from tools.common import wavio  # noqa: E402

FS = 32000
HOP = WIN // 2


def mend_stream(net, coef, var, status, gap, comp, dev, hint=None, refine=False):
    """coef (N,64) の、使えないフレームを MendNet で埋める。ストリームの両端の外は欠落として埋める"""
    n = len(coef)
    pad = WIN // 4
    c = np.concatenate([np.zeros((pad, 64), np.float32), coef, np.zeros((WIN, 64), np.float32)])
    v = np.concatenate([-np.ones((pad, 5), np.float32), var, -np.ones((WIN, 5), np.float32)])
    s = np.concatenate([np.full(pad, 3), status, np.full(WIN, 3)])
    g = np.concatenate([np.zeros(pad, np.float32), gap, np.zeros(WIN, np.float32)])
    hh = np.concatenate([np.zeros((pad, 64), np.float32), hint[0], np.zeros((WIN, 64), np.float32)]) if hint else None
    hq = np.concatenate([np.zeros((pad, 5), np.float32), hint[1], np.zeros((WIN, 5), np.float32)]) if hint else None
    out = coef.copy()
    starts = list(range(0, n + pad - WIN // 4, HOP))
    for a in starts:
        sl = slice(a, a + WIN)
        b = lambda x, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(x[sl])).to(dev, dt)[None]  # noqa: E731
        with torch.no_grad():
            x, sig, usable = build_input(b(c), b(v), b(s, torch.long), b(g), torch.tensor([comp], device=dev), *((b(hh), b(hq)) if hint else ()))
            est = compose(net(x), b(c), sig, usable, refine)[0].cpu().numpy()
        lo, hi = a + WIN // 4, a + WIN // 4 + HOP                       # 窓の中央 50 フレーム (padded 座標)
        for t in range(max(lo, pad), min(hi, pad + n)):
            out[t - pad] = est[t - a]
    return out


def to_audio(m, s):
    mm, ss = haar_inv(m).reshape(-1), haar_inv(s).reshape(-1)
    return np.stack([mm + ss, mm - ss], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--file", type=int, default=0)
    ap.add_argument("--scen", type=int, default=2)
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--clean", action="store_true", help="使えるフレームを原音にする (補間だけを聴く。受信の雑音を除く)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = sorted(glob.glob(os.path.join(a.data, a.profile, "mend", "val", "*.npz")))[a.file]
    z = np.load(f)
    r = slice(int(a.start * FS / 64), int((a.start + a.seconds) * FS / 64))
    truth, dec = z["truth"][r], z["dec"][a.scen][r]
    var, st, gap = z["var"][a.scen][r], z["status"][a.scen][r], z["gap"][a.scen][r]
    if a.clean:
        dec = np.where((st <= 1)[:, None], truth, 0.0).astype(np.float32)
        var = np.where((var >= 0) & (st <= 1)[:, None, None], 1e-9, var).astype(np.float32)
    ck = torch.load(a.ckpt, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    us = st <= 1
    ar, md = [], []
    for ch in range(2):
        d = dec[:, ch * 64:ch * 64 + 64]
        ar.append(ar_fill(d.astype(np.float64), us))
        md.append(mend_stream(net, d, var[:, ch], st, gap, ch, dev, (z['hint_h'][a.scen][r][:, ch * 64:ch * 64 + 64], z['hint_q'][a.scen][r][:, ch]) if ck.get('hint', False) else None, ck.get('refine', False)))
    sets = {"original": [truth[:, :64], truth[:, 64:]], "received": [dec[:, :64], dec[:, 64:]], "ar": ar, "mendnet": md, "mendnet_join": [join_ar(m, us) for m in md]}
    os.makedirs(a.out, exist_ok=True)
    ref = to_audio(*sets["original"])
    bad = np.repeat(~us, 64)
    for name, (m, s) in sets.items():
        y = to_audio(m, s)
        wavio.write(os.path.join(a.out, f"{os.path.basename(f)[:30]}_s{a.scen}{'_clean' if a.clean else ''}_{name}.wav"), y, FS)
        e = ((y - ref)[bad] ** 2).sum()
        print(f"{name:9s} 使えない区間の SNR {10 * np.log10((ref[bad] ** 2).sum() / e) if e > 0 else float('inf'):+6.2f} dB")
    print(f"シナリオ {z['scen'][a.scen]}、使えないフレーム {int((~us).sum())}/{len(us)}、出力 {a.out}")


if __name__ == "__main__":
    main()
