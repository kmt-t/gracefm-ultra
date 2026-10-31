#!/usr/bin/env python3
"""
HaarNet-lite の学習。prep_haarnet.py のデータから、64 フレームの窓をランダムに切って、
損失 = (窓内の既知フレームの誤差エネルギー) ÷ (正解係数のエネルギー + 下限)。欠落フレームは入力を 0 にして損失から除く。

  .venv/bin/python tools/train_haarnet.py --profile B200 --minutes 20 --out runs/hn_b200.pt
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
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.haarnet import WIN, HaarNetLite, features, make_meta  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}


def load(split, profile):
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "data", profile, "hn", split, "*.npz"))):
        d = np.load(f)
        out.append({k: d[k] for k in d.files})
    return out


def batch(data, bs, rng, min_esn0=-1e9):
    L, V, LO, ST, T = [], [], [], [], []
    while len(L) < bs:
        d = data[rng.integers(len(data))]
        s = int(rng.integers(d["llr"].shape[0]))
        if d["scen"][s][0] < min_esn0:                      # ファインチューニング用: 基準 Es/N0 の下限
            continue
        t0 = int(rng.integers(0, d["llr"].shape[1] - WIN))
        lost = d["lost"][s, t0:t0 + WIN]
        if lost.mean() > 0.5:
            continue
        tr, st = d["coef0"][t0:t0 + WIN], d["step"][t0:t0 + WIN]
        if np.mean(tr[:, :4] ** 2) < 100 * np.mean(st[:, :4] ** 2):          # ほぼ無音の窓は除く (M の k=0 が刻みに近い)
            continue
        L.append(d["llr"][s, t0:t0 + WIN].astype(np.float32)); V.append(d["var"][s, t0:t0 + WIN].astype(np.float32))
        LO.append(lost); ST.append(st); T.append(tr)
    f = lambda a: torch.from_numpy(np.stack(a))  # noqa: E731
    return f(L), f(ST), f(V), f(LO), f(T)


def loss_fn(net, meta, llr, step, var, lost, truth, edge=4):
    x, S, e0n = features(llr, step, var, lost, meta)
    est = e0n + net(x)
    t = truth / S
    keep = (~lost).float()[..., None].clone()
    keep[:, :edge] = 0
    keep[:, -edge:] = 0
    w = S ** 2
    nvalid = keep[..., 0].sum(1)
    err = (((est - t) ** 2) * w * keep).sum((1, 2))
    ref = ((t ** 2) * w * keep).sum((1, 2)) + 0.05 * nvalid * w.sum((1, 2)) + 1e-6      # 下限: 観測レベルの -13 dB
    err0 = (((e0n - t) ** 2) * w * keep).sum((1, 2))
    return (err / ref).mean(), (err0 / ref).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="", help="このチェックポイントの重みから始める")
    ap.add_argument("--min-esn0", type=float, default=-1e9, help="この Es/N0 [dB] 以上のシナリオだけで学習する")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "runs", f"hn_{a.profile}.pt")
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(len(CPUS))
    rng = np.random.default_rng(0)
    data = load("train", a.profile)
    meta = make_meta(data[0], PROFILES[a.profile])
    cfg = dict(nbit=meta.nbit, ncoef=meta.ncoef, ch=128, blocks=8)
    net = HaarNetLite(**cfg)
    print(f"{a.profile}: 学習 {len(data)} 音源、bit {meta.nbit}、係数 {meta.ncoef}、パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location="cpu")["model"])
        print(f"{a.resume} から再開 (Es/N0 >= {a.min_esn0})", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    t0, step, run, run0 = time.time(), 0, [], []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        loss, base = loss_fn(net, meta, *batch(data, a.bs, rng, a.min_esn0))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.item()); run0.append(base.item())
        step += 1
        if step % 50 == 0:
            print(f"step {step:5d} {time.time() - t0:5.0f}s  E0 {-10 * np.log10(np.mean(run0[-50:])):6.2f} dB  net {-10 * np.log10(np.mean(run[-50:])):6.2f} dB", flush=True)
        if step % 500 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
