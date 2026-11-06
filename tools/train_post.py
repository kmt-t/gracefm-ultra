#!/usr/bin/env python3
"""
PostNet (MDCT 領域の超解像) の学習。data/<profile>/post/{train,val} (gen_post.py) の、確定した MendNet の出力から切り出す。
切り出しは、損失に入るフレームで超解像をかける判定 (postnet.gate: 元の音質が良いとき) が全部オンのところだけ (文脈はオフでもよい)。切り出しは CROP フレーム = MDCT の F フレーム (合成の文脈に前後 REACH フレームを足す)。先頭 SKIP フレーム (MDCT) は因果の段の文脈なので、損失に入れない。
損失は postnet.post_loss (帯域ごとの相対誤差の対数 + 複数解像度の対数振幅)。

  .venv/bin/python tools/train_post.py --profile B200-L --minutes 120 --out runs/post_B200-L.pt
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
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.transform import REACH  # noqa: E402
from gracefm_ultra.postnet import FRAME, M, PostNet, Transform, gate, post_loss, receptive_past  # noqa: E402

F = 192                                   # MDCT のフレーム数
CROP = (F + 1) * M // FRAME               # フレーム数
SKIP = 96
LEAD = SKIP * M // FRAME                  # 判定を問わない先頭のフレーム数 (文脈だけ。オフの所から入る切り替わり目も学ぶ)
HIGH_HZ = 4000                            # 検証で別に見る高い帯域の下端
assert SKIP >= receptive_past()


class Crops:
    """区間・全シナリオの MendNet の出力と原音を (CPU に) 保持して、損失に入るフレーム (先頭 LEAD より後) で
    超解像をかける判定が全部オンのところを切り出す"""

    def __init__(self, files, power):
        self.e0, self.sd, self.truth, self.status, self.starts = [], [], [], [], []
        self.comps = []
        for f in files:
            z = np.load(f)
            st = []
            for sc in range(z["e0"].shape[0]):
                off = np.concatenate([[0], np.cumsum(~gate(z["e0"][sc], z["sd"][sc], z["status"][sc], power))])
                a = np.flatnonzero(off[CROP:] == off[LEAD:len(off) - CROP + LEAD])
                a = a[(a >= REACH) & (a + CROP + REACH <= len(off) - 1)]
                if len(a):
                    st.append((sc, a))
            if not st:
                continue
            self.e0.append(torch.from_numpy(z["e0"]))
            self.sd.append(torch.from_numpy(z["sd"]))
            self.truth.append(torch.from_numpy(z["truth"]))
            self.status.append(torch.from_numpy(z["status"]))
            self.starts.append(st)
            self.comps = [ch for ch in (0, 1) if z["truth"][:, ch * 64:ch * 64 + 64].any()]   # 送っている成分
        if not self.starts:
            raise SystemExit("判定がオンの切り出しがありません。このプロファイルは閾値 (GATE_ON) に届いていません")

    def locate(self, bs, rng):
        """[(区間, シナリオ, 先頭フレーム, 成分), …]"""
        out = []
        for _ in range(bs):
            si = int(rng.integers(len(self.e0)))
            sc, a = self.starts[si][int(rng.integers(len(self.starts[si])))]
            out.append((si, int(sc), int(a[rng.integers(len(a))]), int(rng.choice(self.comps))))
        return out

    def gather(self, locs, dev):
        """→ (e0, sd, truth (N, REACH + CROP + REACH, 64), comp (N,))"""
        e0, sd, tr = [], [], []
        for si, sc, a, ch in locs:
            sl = slice(ch * 64, ch * 64 + 64)
            t = slice(a - REACH, a + CROP + REACH)
            e0.append(self.e0[si][sc, t, sl])
            sd.append(self.sd[si][sc, t, sl].float())
            tr.append(self.truth[si][t, sl])
        f = lambda v: torch.stack(v).to(dev)  # noqa: E731
        return f(e0), f(sd), f(tr), torch.tensor([x[3] for x in locs], device=dev)


def forward(net, tf, b, high):
    """→ (損失 (N,), 二乗和 (推定の誤差、入力の誤差、原音) × (全帯域、ビン high 以上) (6,))。二乗和は先頭 SKIP フレームを除く"""
    e0, sd, truth, comp = b
    X, var, T = tf.mdct(tf.wave(e0, REACH)), tf.mdct_var(tf.wave_var(sd, REACH)), tf.mdct(tf.wave(truth, REACH))
    Y = net(X, var, comp)
    loss = post_loss(tf, Y, T, SKIP)
    sq = lambda v, a: (v[:, a:, SKIP:] ** 2).sum().item()  # noqa: E731
    return loss, np.array([sq(v, a) for a in (0, high) for v in (Y - T, X - T, T)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200-L")
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
    tf = Transform(PROFILES[a.profile].transform).to(dev)
    high = int(HIGH_HZ * 2 * M / PROFILES[a.profile].fs)
    tr = Crops(sorted(glob.glob(os.path.join(d, "train", "*.npz"))), tf.power)
    va = Crops(sorted(glob.glob(os.path.join(d, "val", "*.npz"))), tf.power)
    cfg = dict(ch=256)
    net = PostNet(**cfg).to(dev)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev)["model"])
    nsc = lambda c: sum(len(x) for x in c.starts)  # noqa: E731
    print(f"区間 学習 {len(tr.e0)} (シナリオ {nsc(tr)})、検証 {len(va.e0)} (シナリオ {nsc(va)})、パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    vrng = np.random.default_rng(1)
    vb = [va.gather(va.locate(a.bs, vrng), dev) for _ in range(16)]
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        loss = forward(net, tf, tr.gather(tr.locate(a.bs, rng), dev), high)[0]
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.mean().item())
        step += 1
        if step % 200 == 0:
            with torch.no_grad():
                net.eval()
                v = [forward(net, tf, b, high) for b in vb]
                net.train()
            vl = np.mean([x[0].mean().item() for x in v])
            e, e_in, t2, he, he_in, ht2 = sum(x[1] for x in v)
            db = lambda t, e: 10 * np.log10(t / e)  # noqa: E731
            print(f"step {step:6d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-200:]):.4f}  検証 loss {vl:.4f}  SNR 入力 {db(t2, e_in):+.2f} → {db(t2, e):+.2f} dB"
                  f"  4 kHz 以上 {db(ht2, he_in):+.2f} → {db(ht2, he):+.2f} dB", flush=True)
        if step % 1000 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
