#!/usr/bin/env python3
"""
MendHaarNet の後段 (HaarStage) の学習。前段 MendNet (tools/train_mendbits.py) は固定。窓の両成分に前段を走らせ、その LLR と刻みから、
Haar 係数と怪しさを出す。損失は成分ごとの相対二乗誤差 + 怪しさの NLL。

  .venv/bin/python tools/train_mendhaar.py --profile B200 --mend runs/mendbits_B200.pt --minutes 30 --out runs/mendhaar_B200.pt
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
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.haarnet import make_meta  # noqa: E402
from gracefm_ultra.mendhaar import HaarStage, estimate, run_mend, stage_loss  # noqa: E402
from gracefm_ultra.mendnet import Q22, MendNet  # noqa: E402
from train_mend import CPUS, Windows  # noqa: E402


def load_front(path, dev):
    ck = torch.load(path, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def step_batch(front, win, locs, meta, nbit):
    """窓 locs の、前段の出力と真の係数。→ (llr, step, status (B,T), truth (B,T,128) 満点 2^22 の尺度)"""
    per = [win.gather(locs, [ch] * len(locs)) for ch in range(2)]
    with torch.no_grad():
        llr, step = run_mend(front, per, win.bit_idx.to(win.device), win.bit_ok.to(win.device), nbit)
    return llr, step, per[0]["status"], torch.cat([per[0]["target"], per[1]["target"]], -1) * Q22


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--mend", required=True, help="前段のチェックポイント")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "runs", f"mendhaar_{a.profile}.pt")
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = os.path.join(a.data, a.profile, "mend")
    tr = Windows(sorted(glob.glob(os.path.join(d, "train", "*.npz"))), dev, bits=True)
    va = Windows(sorted(glob.glob(os.path.join(d, "val", "*.npz"))), dev, bits=True)
    z = np.load(sorted(glob.glob(os.path.join(d, "train", "*.npz")))[0])
    meta = make_meta({k: z[k] for k in ("pos", "coef_of_bit", "group")}, PROFILES[a.profile]).to(dev)
    nbit = meta.nbit
    front = load_front(a.mend, dev)
    cfg = dict(nbit=nbit, ncoef=meta.ncoef, ngroup=meta.G.shape[1], ch=128, blocks=8)
    net = HaarStage(**cfg).to(dev)
    print(f"パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    rng, vrng = np.random.default_rng(0), np.random.default_rng(1)
    vlocs = [va.locate(a.bs, vrng, True) for _ in range(8)]
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        llr, st, status, truth = step_batch(front, tr, tr.locate(a.bs, rng, True), meta, nbit)
        est, S, e0, logvar = estimate(net, llr, st, status, meta)
        loss, _, _ = stage_loss(est, e0, S, logvar, truth, meta)
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.mean().item())
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                net.eval()
                v = []
                for locs in vlocs:
                    llr, st, status, truth = step_batch(front, va, locs, meta, nbit)
                    est, S, e0, logvar = estimate(net, llr, st, status, meta)
                    v.append(torch.stack(stage_loss(est, e0, S, logvar, truth, meta)[1:]).mean(1))
                net.train()
            v = torch.stack(v).mean(0).cpu().numpy()
            print(f"step {step:5d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-100:]):.4f}  検証 相対誤差 前段だけ {-10 * np.log10(v[0]):.2f} dB  後段 {-10 * np.log10(v[1]):.2f} dB", flush=True)
        if step % 500 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "mend": a.mend, "step": step}, out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "mend": a.mend, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
