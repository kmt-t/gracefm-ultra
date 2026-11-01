#!/usr/bin/env python3
"""
MendNet の学習。data/<profile>/mend/{train,val} (gen_hn.py --mend) の、実際の復号の結果から窓 (WIN フレーム) を切り出す。
窓は、使えない (壊れた・欠落) フレームを含むものを選ぶ。損失は使えないフレームの帯域ごとの相対二乗誤差 + 接続部の変化量の誤差。

  .venv/bin/python tools/train_mend.py --profile B200 --minutes 30 --out runs/mend_B200.pt
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.mendnet import C_IN, C_IN_HINT, WIN, MendNet, build_input, mend_loss  # noqa: E402

CPUS = {int(v) for v in os.environ.get("ULTRA_CPUS", "0,1,5,6,7,11").split(",")}
MIN_USABLE = 10


class Windows:
    """全区間・全シナリオを (CPU に) 保持して、窓をバッチで切り出す"""

    def __init__(self, files, device):
        self.truth, self.dec, self.var, self.status, self.gap, self.hint_h, self.hint_q = [], [], [], [], [], [], []
        self.bad = []                                              # (区間, シナリオ, フレーム) の、使えないフレーム
        for si, f in enumerate(files):
            z = np.load(f)
            self.truth.append(torch.from_numpy(z["truth"])
            self.dec.append(torch.from_numpy(z["dec"])
            self.var.append(torch.from_numpy(z["var"])
            self.status.append(torch.from_numpy(z["status"])
            self.gap.append(torch.from_numpy(z["gap"])
            self.hint_h.append(torch.from_numpy(z["hint_h"])
            self.hint_q.append(torch.from_numpy(z["hint_q"])
            s, fr = np.nonzero(z["status"] >= 2)
            self.bad.append(np.stack([np.full_like(s, si), s, fr], 1))
        self.bad = np.concatenate(self.bad)
        self.ncomp = self.var[0].shape[2]
        self.device = device

    def batch(self, bs, rng):
        out = {k: [] for k in ("coef", "var", "status", "gap", "comp", "target", "hint_h", "hint_q")}
        while len(out["coef"]) < bs:
            si, sc, fr = self.bad[rng.integers(len(self.bad))]
            n = self.truth[si].shape[0]
            a = int(np.clip(fr - rng.integers(0, WIN), 0, n - WIN))
            st = self.status[si][sc, a:a + WIN]
            if (st <= 1).sum() < MIN_USABLE:
                continue
            ch = int(rng.integers(self.ncomp))
            sl = slice(ch * 64, ch * 64 + 64)
            out["coef"].append(self.dec[si][sc, a:a + WIN, sl])
            out["target"].append(self.truth[si][a:a + WIN, sl])
            out["var"].append(self.var[si][sc, a:a + WIN, ch])
            out["status"].append(st)
            out["gap"].append(self.gap[si][sc, a:a + WIN])
            out["hint_h"].append(self.hint_h[si][sc, a:a + WIN, sl])
            out["hint_q"].append(self.hint_q[si][sc, a:a + WIN, ch])
            out["comp"].append(torch.tensor(ch))
        return {k: torch.stack(v).to(self.device) for k, v in out.items()}


def forward(net, b, hint):
    x, sig, usable = build_input(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], *((b["hint_h"], b["hint_q"]) if hint else ()))
    y = net(x)
    return mend_loss(y, b["target"], sig, usable, b["coef"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", default="", help="このチェックポイントの重みから始める")
    ap.add_argument("--hint", action="store_true", help="v2: 壊れたフレームの手がかりを入力に足す")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "runs", f"mend_{a.profile}.pt")
    os.sched_setaffinity(0, CPUS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    d = os.path.join(a.data, a.profile, "mend")
    tr = Windows(sorted(glob.glob(os.path.join(d, "train", "*.npz"))), dev)
    va = Windows(sorted(glob.glob(os.path.join(d, "val", "*.npz"))), dev)
    print(f"使えないフレーム 学習 {len(tr.bad)}、検証 {len(va.bad)}", flush=True)
    cfg = dict(c_in=C_IN_HINT if a.hint else C_IN, ch=192, blocks=12)
    net = MendNet(**cfg).to(dev)
    print(f"パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev)["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    vrng = np.random.default_rng(1)
    vb = [va.batch(a.bs, vrng) for _ in range(8)]
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        loss, _ = forward(net, tr.batch(a.bs, rng), a.hint)
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.mean().item())
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                net.eval()
                v = torch.cat([forward(net, b, a.hint)[1] for b in vb]).mean(0).cpu().numpy()
                net.train()
            print(f"step {step:5d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-100:]):.4f}  検証 帯域別 {np.round(v, 3)}", flush=True)
        if step % 500 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "hint": a.hint, "step": step}, out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "hint": a.hint, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
