#!/usr/bin/env python3
"""
PostNet (MDCT 領域の超解像) の学習。data/<profile>/post/{train,val} (gen_post.py) の、確定した MendNet の出力から切り出す。
切り出しは CROP フレーム (Haar) = MDCT の F フレーム。先頭 SKIP フレーム (MDCT) は因果の段の文脈なので、損失に入れない。
損失は postnet.post_loss (MDCT の log(1 + 相対誤差) + 複数解像度の対数振幅)。

  .venv/bin/python tools/train_post.py --profile B200 --minutes 120 --out runs/post_B200.pt
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
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm_ultra.postnet import FRAME, M, PostNet, Transform, post_loss, receptive_past  # noqa: E402

F = 192                                   # MDCT のフレーム数
CROP = (F + 1) * M // FRAME               # Haar のフレーム数
SKIP = 96
assert SKIP >= receptive_past()


class Crops:
    """区間・全シナリオの MendNet の出力と原音を (CPU に) 保持して、切り出す"""

    def __init__(self, files):
        self.e0, self.sd, self.truth, self.status = [], [], [], []
        for f in files:
            z = np.load(f)
            self.e0.append(torch.from_numpy(z["e0"]))
            self.sd.append(torch.from_numpy(z["sd"]))
            self.truth.append(torch.from_numpy(z["truth"]))
            self.status.append(torch.from_numpy(z["status"]))

    def locate(self, bs, rng):
        """[(区間, シナリオ, 先頭フレーム, 成分), …]"""
        out = []
        for _ in range(bs):
            si = int(rng.integers(len(self.e0)))
            S, T = self.e0[si].shape[:2]
            out.append((si, int(rng.integers(S)), int(rng.integers(T - CROP + 1)), int(rng.integers(2))))
        return out

    def gather(self, locs, dev):
        """→ (e0, sd, truth (N, CROP, 64), status (N, CROP), comp (N,))"""
        e0, sd, tr, st = [], [], [], []
        for si, sc, a, ch in locs:
            sl = slice(ch * 64, ch * 64 + 64)
            e0.append(self.e0[si][sc, a:a + CROP, sl])
            sd.append(self.sd[si][sc, a:a + CROP, sl].float())
            tr.append(self.truth[si][a:a + CROP, sl])
            st.append(self.status[si][sc, a:a + CROP])
        f = lambda v: torch.stack(v).to(dev)  # noqa: E731
        return f(e0), f(sd), f(tr), f(st), torch.tensor([x[3] for x in locs], device=dev)


def forward(net, tf, b):
    """→ (損失 (N,), 二乗和 (推定の誤差、入力の誤差、原音) (3,))。二乗和は先頭 SKIP フレームを除く"""
    e0, sd, truth, _, comp = b
    X, var, T = tf.mdct(tf.wave(e0)), tf.mdct_var(tf.wave_var(sd)), tf.mdct(tf.wave(truth))
    Y = net(X, var, comp)
    loss, _ = post_loss(tf, Y, T, SKIP)
    sq = lambda v: (v[:, :, SKIP:] ** 2).sum().item()  # noqa: E731
    return loss, np.array([sq(Y - T), sq(X - T), sq(T)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", default="", help="このチェックポイントの重みから始める")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "runs", f"post_{a.profile}.pt")
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    d = os.path.join(a.data, a.profile, "post")
    tr = Crops(sorted(glob.glob(os.path.join(d, "train", "*.npz"))))
    va = Crops(sorted(glob.glob(os.path.join(d, "val", "*.npz"))))
    tf = Transform().to(dev)
    cfg = dict(ch=256)
    net = PostNet(**cfg).to(dev)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev)["model"])
    print(f"区間 学習 {len(tr.e0)}、検証 {len(va.e0)}、パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    vrng = np.random.default_rng(1)
    vb = [va.gather(va.locate(a.bs, vrng), dev) for _ in range(16)]
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        loss = forward(net, tf, tr.gather(tr.locate(a.bs, rng), dev))[0]
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.mean().item())
        step += 1
        if step % 200 == 0:
            with torch.no_grad():
                net.eval()
                v = [forward(net, tf, b) for b in vb]
                net.train()
            e, e_in, t2 = sum(x[1] for x in v)
            print(f"step {step:6d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-200:]):.4f}  検証 SNR 入力 {10 * np.log10(t2 / e_in):+.2f} dB → {10 * np.log10(t2 / e):+.2f} dB",
                  flush=True)
        if step % 1000 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
