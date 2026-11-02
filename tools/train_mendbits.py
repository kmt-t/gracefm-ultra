#!/usr/bin/env python3
"""
MendNet (ビット出力版) の学習。復号した係数・状態・hint と受信 LLR から、全フレームの仮数ビットの LLR と刻み (ヘッダ) を出す。
損失は、真のビットに対する BCE + 刻み (log2) の Huber。使えるフレームと使えないフレームを別々に平均する。
data/<profile>/mend/{train,val} (gen_hn.py --mend) を使う。

  .venv/bin/python tools/train_mendbits.py --profile B200 --minutes 30 --out runs/mendbits_B200.pt
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.mendnet import C_IN_HINT, MendNet, bits_loss, bits_out, build_input_bits  # noqa: E402
from train_mend import CPUS, Windows  # noqa: E402


def forward(net, b):
    x, _, usable, sf = build_input_bits(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], b["hint_h"], b["hint_q"], b["llr"], b["dstep"])
    llr, step = bits_out(net(x), b["llr"], sf, b["dstep"])
    return llr, step, bits_loss(llr, step, b, usable), usable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "runs", f"mendbits_{a.profile}.pt")
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    d = os.path.join(a.data, a.profile, "mend")
    tr = Windows(sorted(glob.glob(os.path.join(d, "train", "*.npz"))), dev, bits=True)
    va = Windows(sorted(glob.glob(os.path.join(d, "val", "*.npz"))), dev, bits=True)
    nb = tr.bit_idx.shape[1]
    cfg = dict(c_in=C_IN_HINT + nb + 128, ch=192, blocks=12, c_out=nb + 64)
    net = MendNet(**cfg).to(dev)
    print(f"使えないフレーム 学習 {len(tr.bad)}、検証 {len(va.bad)}、ビット {nb}、パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    vrng = np.random.default_rng(1)
    vb = [va.batch(a.bs, vrng, anywhere=True) for _ in range(8)]
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        loss = forward(net, tr.batch(a.bs, rng, True))[2][0]
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.mean().item())
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                net.eval()
                v = [forward(net, b)[2] for b in vb]
                net.train()
            bad, good = (torch.cat([x[1] for x in v]).mean().item(), torch.cat([x[2] for x in v]).mean().item())
            print(f"step {step:5d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-100:]):.4f}  検証 BCE 使えない {bad:.4f}  使える {good:.4f}", flush=True)
        if step % 500 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
