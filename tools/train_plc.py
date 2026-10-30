#!/usr/bin/env python3
"""
PlcNet の学習。学習用音源の M 成分 (32 kHz) を Haar 係数にして、200 ms の窓にランダムな欠落 (1〜3 か所、1〜25 フレーム) を作り、
欠落フレームの係数を予測させる。損失は欠落部の誤差エネルギー ÷ 窓の既知部の信号電力 (係数領域 = 時間領域)。
文脈には、ビット誤り相当の雑音 (信号比 20〜60 dB) を加える。

  .venv/bin/python tools/train_plc.py --minutes 20 --out runs/plc.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.plcnet import BAND, BLOCK, WIN, PlcNet, haar_fwd, prepare  # noqa: E402
from tools.common import wavio  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}
GAPS = [1, 2, 3, 5, 8, 12, 18, 25]


def load_train(src, seconds):
    files = sorted(f for f in os.listdir(src) if f.endswith((".flac", ".wav")))
    files = [f for i, f in enumerate(files) if i % 7 != 3]                   # 検証用 (i % 7 == 3) は使わない
    out = []
    for f in files:
        x, fs = wavio.read(os.path.join(src, f))
        x = wavio.resample(x, fs, 32000)
        m = (x * 0.5).mean(1)
        st = len(m) // 4
        m = m[st:st + int(seconds * 32000)]
        n = len(m) // BLOCK
        out.append(haar_fwd(m[:n * BLOCK].reshape(n, BLOCK) * (1 << 23)).astype(np.float32))
    return out


def make_batch(coefs, bs, rng):
    X = np.zeros((bs, 65, WIN), np.float32)
    T = np.zeros((bs, 64, WIN), np.float32)
    M = np.zeros((bs, WIN), np.float32)
    S = np.zeros((bs, 5), np.float32)
    n = 0
    while n < bs:
        c = coefs[rng.integers(len(coefs))]
        s = int(rng.integers(0, len(c) - WIN))
        w = c[s:s + WIN].astype(np.float64)
        if np.mean(w ** 2) < 1e-6 * (1 << 40):
            continue
        known = np.ones(WIN, bool)
        for _ in range(int(rng.integers(1, 4))):
            g = int(rng.choice(GAPS))
            a = int(rng.integers(0, WIN - g + 1))
            known[a:a + g] = False
        if known.sum() < 10 or known.all():
            continue
        noisy = w.copy()
        db = rng.uniform(20, 60)
        sg = np.array([np.sqrt(np.mean(w[known][:, BAND == b] ** 2)) for b in range(5)])
        noisy += rng.normal(size=w.shape) * (sg[BAND] * 10 ** (-db / 20))[None, :]
        inp, sig = prepare(noisy, known)
        X[n], S[n], M[n] = inp, sig, (~known)
        T[n] = (w / sig[BAND][None, :]).T
        n += 1
    return torch.from_numpy(X), torch.from_numpy(T), torch.from_numpy(M), torch.from_numpy(S)


def loss_fn(net, X, T, M, S):
    Y = net(X)
    wband = torch.from_numpy(np.eye(5)[BAND].T).float()          # (5, 64)
    w = (S ** 2) @ wband                                         # (B, 64) 帯域の電力を係数に配る
    err = ((Y - T) ** 2 * w[:, :, None] * M[:, None, :]).sum((1, 2))
    ref = (w.sum(1) * M.sum(1)).clamp_min(1.0)                   # 既知部の平均電力 × 欠落フレーム数
    return (err / ref).mean(), (err / ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(GRACEFM, "eval_audio", "train_lossless"))
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seconds", type=float, default=150, help="音源ごとに使う長さ")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "plc.pt"))
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(len(CPUS))
    rng = np.random.default_rng(0)
    coefs = load_train(a.src, a.seconds)
    print(f"学習音源 {len(coefs)} 本、{sum(len(c) for c in coefs) * BLOCK / 32000 / 60:.1f} 分", flush=True)
    cfg = dict(ch=192, blocks=12)
    net = PlcNet(**cfg)
    print(f"パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        X, T, M, S = make_batch(coefs, a.bs, rng)
        loss, _ = loss_fn(net, X, T, M, S)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.item())
        step += 1
        if step % 50 == 0:
            print(f"step {step:5d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-50:]):.4f}  ({-10 * np.log10(np.mean(run[-50:])):+.2f} dB 相当)", flush=True)
        if step % 500 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, a.out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "step": step}, a.out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
