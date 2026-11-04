#!/usr/bin/env python3
"""
PostNet (MDCT 領域の超解像) の評価。各シナリオを丸ごと (約 20 秒) 処理し、MendNet の出力 (PostNet の入力) と比べる。
- 既定: data/<profile>/post/val で、使えないフレームの SNR [dB] を使えない連続区間の長さごとに、使えるフレームの SNR (時間領域、M と S の合計)。
- --awgn: data/<profile>/post/awgn_val で、音声 (ステレオ 32 kHz) の SNR (全体)、PSNR (区間の平均)、ViSQOL (区間の平均) を Es/N0 ごとに。
  復号のまま (data/<profile>/mend_awgn/val の dec) も並べる。

  .venv/bin/python tools/eval_post.py --ckpt runs/post_B200.pt
  .venv/bin/python tools/eval_post.py --ckpt runs/post_B200.pt --awgn
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm_ultra.postnet import FRAME, M, PostNet, Transform  # noqa: E402
from eval_mendnet import BUCKETS, CN_OFFSET, runs_of, visqol  # noqa: E402


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    net = PostNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net


def process(net, tf, e0, sd, dev):
    """1 シナリオ e0, sd (T,128) → {"MendNet": 波形 (2, T×64) M/S, "PostNet": 同}"""
    with torch.no_grad():
        c = torch.from_numpy(np.ascontiguousarray(e0.reshape(-1, 2, 64).transpose(1, 0, 2))).to(dev)
        s = torch.from_numpy(np.ascontiguousarray(sd.astype(np.float32).reshape(-1, 2, 64).transpose(1, 0, 2))).to(dev)
        pad = lambda v: torch.nn.functional.pad(v, (M, M))  # noqa: E731
        x, v = tf.wave(c), tf.wave_var(s)
        Y = net(tf.mdct(pad(x)), tf.mdct_var(pad(v)), torch.arange(2, device=dev))
        y = tf.imdct(Y)[:, M:-M]
    return {"MendNet": x.cpu().numpy(), "PostNet": y.cpu().numpy()}


def wave_of(coef, tf):
    """係数 (T,128) → 波形 (2, T×64) M/S"""
    c = torch.from_numpy(np.ascontiguousarray(coef.reshape(-1, 2, 64).transpose(1, 0, 2))).float()
    return tf.wave(c).numpy()


def eval_gaps(files, net, tf, dev):
    methods = ["MendNet", "PostNet"]
    acc, keep = {}, {m: [0.0, 0.0] for m in methods}
    tfc = Transform()
    for f in files:
        z = np.load(f)
        ref = wave_of(z["truth"], tfc)
        rf = (ref ** 2).reshape(2, -1, FRAME).sum((0, 2))
        for sc in range(z["e0"].shape[0]):
            out = process(net, tf, z["e0"][sc], z["sd"][sc], dev)
            us = z["status"][sc] <= 1
            run = runs_of(us)
            for k in methods:
                ef = ((out[k] - ref) ** 2).reshape(2, -1, FRAME).sum((0, 2))
                keep[k][0] += rf[us].sum()
                keep[k][1] += ef[us].sum()
                for name, lo, hi in BUCKETS:
                    m = (~us) & (run >= lo) & (run <= hi)
                    s = acc.setdefault((name, k), [0.0, 0.0])
                    s[0] += rf[m].sum()
                    s[1] += ef[m].sum()
        print(f"  {os.path.basename(f)}", flush=True)
    for name, _, _ in BUCKETS:
        print(f"{name:10s}", "  ".join(f"{k} {10 * np.log10(acc[(name, k)][0] / max(acc[(name, k)][1], 1e-30)):+6.2f} dB" for k in methods))
    print("使えるフレーム", "  ".join(f"{k} {10 * np.log10(v[0] / v[1]):+6.2f} dB" for k, v in keep.items()))


def lr(w):
    """M/S (2, N) → L/R (N, 2)"""
    return np.stack([w[0] + w[1], w[0] - w[1]], 1)


def eval_awgn(files, src, net, tf, dev):
    methods = ["復号のまま", "MendNet", "PostNet"]
    acc = {}
    tfc = Transform()
    with Pool(len(CPUS), initializer=pin) as pool:
        for si, f in enumerate(files):
            z = np.load(f)
            dec = np.load(os.path.join(src, os.path.basename(f)))["dec"]
            ref = lr(wave_of(z["truth"], tfc))
            jobs, keys = [], []
            for sc, e0 in enumerate(z["scen"][:, 0]):
                out = process(net, tf, z["e0"][sc], z["sd"][sc], dev)
                out["復号のまま"] = wave_of(dec[sc], tfc)
                for k in methods:
                    y = lr(out[k])
                    e = float(((y - ref) ** 2).sum())
                    s = acc.setdefault(float(e0), {}).setdefault(k, [0.0, 0.0, [], []])
                    s[0] += float((ref ** 2).sum())
                    s[1] += e
                    s[3].append(10 * np.log10(ref.size / max(e, 1e-20)))
                    jobs.append((ref, y))
                    keys.append((float(e0), k))
            for (e0, k), mos in zip(keys, pool.map(visqol, jobs)):
                acc[e0][k][2].append(mos)
            print(f"[{si + 1}/{len(files)}] {os.path.basename(f)}", flush=True)
    print("\nSNR [dB] (全体)、PSNR [dB] (区間の平均)、ViSQOL (区間の平均)")
    for e0, d in sorted(acc.items()):
        print(f"\nEs/N0 {e0:g} dB (C/N0 {e0 + CN_OFFSET:g} dB-Hz)、区間 {len(d[methods[0]][2])}")
        for k in methods:
            s = d[k]
            print(f"  {k:10s} SNR {10 * np.log10(s[0] / s[1]):+6.2f}  PSNR {np.mean(s[3]):6.2f}  ViSQOL {np.mean(s[2]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--awgn", action="store_true")
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load(a.ckpt, dev)
    tf = Transform().to(dev)
    d = os.path.join(a.data, a.profile)
    if a.awgn:
        eval_awgn(sorted(glob.glob(os.path.join(d, "post", "awgn_val", "*.npz"))), os.path.join(d, "mend_awgn", "val"), net, tf, dev)
    else:
        eval_gaps(sorted(glob.glob(os.path.join(d, "post", "val", "*.npz"))), net, tf, dev)


if __name__ == "__main__":
    main()
