#!/usr/bin/env python3
"""
MendHaarNet の学習。前段 MendNet (tools/train_mendbits.py) と後段 HaarStage を、交互に学習する。
- --train back: 前段を固定して後段を学習する。窓の両成分に前段を走らせ、その LLR と刻みから Haar 係数と怪しさを出す。
  損失は成分ごとの相対二乗誤差 + 怪しさの NLL。
- --train front: 後段 (--back) を固定して前段を学習する。損失は前段の損失 (BCE + 刻み) + 後段の損失。後段の誤差が前段にも流れる。

  .venv/bin/python tools/train_mendhaar.py --profile B200 --mend runs/mendbits_B200.pt --minutes 30 --out runs/mendhaar_B200.pt
  .venv/bin/python tools/train_mendhaar.py --profile B200 --train front --mend runs/mendbits_B200.pt --back runs/mendhaar_B200.pt --out runs/mendbits_B200_2.pt
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


def load_front(path, dev, train=False):
    ck = torch.load(path, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev)
    net.load_state_dict(ck["model"])
    net.train(train)
    for p in net.parameters():
        p.requires_grad_(train)
    return net, ck["cfg"]


def step_batch(front, win, locs, nbit, grad=False):
    """窓 locs の、前段の出力と真の係数。→ (llr, step, status (B,T), truth (B,T,128) 満点 2^22 の尺度, 前段の損失 (B,))"""
    per = [win.gather(locs, [ch] * len(locs)) for ch in range(2)]
    with torch.set_grad_enabled(grad):
        llr, step, front_loss = run_mend(front, per, win.bit_idx.to(win.device), win.bit_ok.to(win.device), nbit)
    return llr, step, per[0]["status"], torch.cat([per[0]["target"], per[1]["target"]], -1) * Q22, front_loss


def save(learn, front_train, fcfg, cfg, mend, step, out):
    """前段は train_mendbits.py と同じ形式 (model、cfg)、後段は HaarStage の重みと、前段のチェックポイントの場所"""
    if front_train:
        torch.save({"model": learn.state_dict(), "cfg": fcfg, "step": step}, out)
    else:
        torch.save({"model": learn.state_dict(), "cfg": cfg, "mend": mend, "step": step}, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--train", choices=["back", "front"], default="back", help="学習する段")
    ap.add_argument("--mend", required=True, help="前段のチェックポイント")
    ap.add_argument("--back", default="", help="後段のチェックポイント (--train front では必須。--train back では、あれば重みの初期値)")
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
    front_train = a.train == "front"
    front, fcfg = load_front(a.mend, dev, front_train)
    cfg = dict(nbit=nbit, ncoef=meta.ncoef, ngroup=meta.G.shape[1], ch=128, blocks=8)
    net = HaarStage(**cfg).to(dev)
    if a.back:
        net.load_state_dict(torch.load(a.back, map_location=dev)["model"])
    net.train(not front_train)
    for p in net.parameters():
        p.requires_grad_(not front_train)
    learn = front if front_train else net
    print(f"学習する段 {a.train}、パラメータ {sum(p.numel() for p in learn.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(learn.parameters(), lr=a.lr, weight_decay=1e-4)
    rng, vrng = np.random.default_rng(0), np.random.default_rng(1)
    vlocs = [va.locate(a.bs, vrng, True) for _ in range(8)]
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        llr, st, status, truth, fl = step_batch(front, tr, tr.locate(a.bs, rng, True), nbit, front_train)
        est, S, e0, logvar = estimate(net, llr, st, status, meta)
        loss, _, _ = stage_loss(est, e0, S, logvar, truth, meta)
        if front_train:
            loss = loss + fl
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(learn.parameters(), 1.0)
        opt.step()
        run.append(loss.mean().item())
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                learn.eval()
                v = []
                for locs in vlocs:
                    llr, st, status, truth, _ = step_batch(front, va, locs, nbit)
                    est, S, e0, logvar = estimate(net, llr, st, status, meta)
                    v.append(torch.stack(stage_loss(est, e0, S, logvar, truth, meta)[1:]).mean(1))
                learn.train()
            v = torch.stack(v).mean(0).cpu().numpy()
            print(f"step {step:5d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-100:]):.4f}  検証 相対誤差 前段だけ {-10 * np.log10(v[0]):.2f} dB  後段 {-10 * np.log10(v[1]):.2f} dB", flush=True)
        if step % 500 == 0:
            save(learn, front_train, fcfg, cfg, a.mend, step, out)
    save(learn, front_train, fcfg, cfg, a.mend, step, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
