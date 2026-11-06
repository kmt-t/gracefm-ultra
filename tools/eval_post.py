#!/usr/bin/env python3
"""
PostNet (MDCT 領域の超解像) の評価。各シナリオを丸ごと (約 20 秒) 処理し、MendNet の出力 (PostNet の入力) と比べる。
PostNet の出力は、超解像をかける判定 (postnet.gate) がオフの MDCT フレームでは MendNet の出力のまま。
- 既定: data/<profile>/post/val の全シナリオの、判定がオンのフレームで、帯域ごとの SNR、エネルギーの比、対数スペクトル距離 (MDCT 領域、M と S の合計)。
- --awgn: data/<profile>/post/awgn_val で、音声 (ステレオ 32 kHz) の SNR (全体)、PSNR (区間の平均)、ViSQOL (区間の平均) を Es/N0 ごとに。
  復号のまま (data/<profile>/mend_awgn/val の dec) も並べる。

  .venv/bin/python tools/eval_post.py --ckpt runs/post_B200.pt
  .venv/bin/python tools/eval_post.py --ckpt runs/post_B200.pt --awgn
  .venv/bin/python tools/eval_post.py --ckpt runs/post_B200.pt --match Chopin,Scarlatti,Kimiko   # ピアノだけ
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
from gracefm_ultra.postnet import FRAME, M, PostNet, Transform, gate, mdct_gate  # noqa: E402
from eval_mendnet import CN_OFFSET, visqol  # noqa: E402


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    net = PostNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net


def process(net, tf, e0, sd, status, dev):
    """1 シナリオ e0, sd (T,128)、status (T,) → ({"MendNet": 波形 (2, T×64) M/S, "PostNet": 同}, 判定 (T,))"""
    g = gate(e0, sd, status)
    with torch.no_grad():
        c = torch.from_numpy(np.ascontiguousarray(e0.reshape(-1, 2, 64).transpose(1, 0, 2))).to(dev)
        s = torch.from_numpy(np.ascontiguousarray(sd.astype(np.float32).reshape(-1, 2, 64).transpose(1, 0, 2))).to(dev)
        pad = lambda v: torch.nn.functional.pad(v, (M, M))  # noqa: E731
        x, v = tf.wave(c), tf.wave_var(s)
        X = tf.mdct(pad(x))
        Y = net(X, tf.mdct_var(pad(v)), torch.arange(2, device=dev))
        on = torch.from_numpy(mdct_gate(g, X.shape[2], M // FRAME)).to(dev)
        y = tf.imdct(torch.where(on, Y, X))[:, M:-M]
    return {"MendNet": x.cpu().numpy(), "PostNet": y.cpu().numpy()}, g


def wave_of(coef, tf):
    """係数 (T,128) → 波形 (2, T×64) M/S"""
    c = torch.from_numpy(np.ascontiguousarray(coef.reshape(-1, 2, 64).transpose(1, 0, 2))).float()
    return tf.wave(c).numpy()


BANDS = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000)]   # [Hz]
LSD_FLOOR = 1e-10                                                                   # 対数スペクトル距離のパワーの下限 (ビン 1 個あたり)


def eval_bands(files, net, tf, dev):
    """帯域ごとに、SNR、出力と原音のエネルギーの比、対数スペクトル距離 (MDCT のビン×フレームの RMS) を、M と S の合計で"""
    methods = ["MendNet", "PostNet"]
    hz = (np.arange(M) + 0.5) * 16000 / M
    acc = {(b, k): [0.0, 0.0, 0.0, 0.0, 0] for b in BANDS for k in methods}   # 原音、誤差、出力、LSD の二乗和、数
    n_on = n_all = 0
    tfc = Transform()
    for f in files:
        z = np.load(f)
        R = tfc.mdct(torch.from_numpy(wave_of(z["truth"], tfc))).numpy()
        for sc in range(z["e0"].shape[0]):
            out, g = process(net, tf, z["e0"][sc], z["sd"][sc], z["status"][sc], dev)
            on = mdct_gate(g, R.shape[2], 0)
            n_on, n_all = n_on + on.sum(), n_all + on.size
            if not on.any():
                continue
            for k in methods:
                Y = tfc.mdct(torch.from_numpy(out[k])).numpy()
                for b in BANDS:
                    sel = (hz >= b[0]) & (hz < b[1])
                    r, y = R[:, sel][:, :, on], Y[:, sel][:, :, on]
                    d = 10 * np.log10((y ** 2 + LSD_FLOOR) / (r ** 2 + LSD_FLOOR))
                    s = acc[(b, k)]
                    s[0] += (r ** 2).sum()
                    s[1] += ((y - r) ** 2).sum()
                    s[2] += (y ** 2).sum()
                    s[3] += (d ** 2).sum()
                    s[4] += d.size
        print(f"  {os.path.basename(f)}", flush=True)
    print(f"判定がオンのフレーム {n_on / n_all:.1%}")
    print("帯域 [kHz]    SNR [dB] / エネルギー比 [dB] / 対数スペクトル距離 [dB]")
    for b in BANDS:
        cols = []
        for k in methods:
            s = acc[(b, k)]
            cols.append(f"{k} {10 * np.log10(s[0] / s[1]):+6.2f} / {10 * np.log10(s[2] / s[0]):+5.2f} / {np.sqrt(s[3] / s[4]):5.2f}")
        print(f"{b[0] // 1000}-{b[1] // 1000:<3d}" + "   ".join(cols))


def lr(w):
    """M/S (2, N) → L/R (N, 2)"""
    return np.stack([w[0] + w[1], w[0] - w[1]], 1)


def eval_awgn(files, src, net, tf, dev, esn0=None):
    methods = ["復号のまま", "MendNet", "PostNet"]
    acc, on = {}, {}
    tfc = Transform()
    with Pool(len(CPUS), initializer=pin) as pool:
        for si, f in enumerate(files):
            z = np.load(f)
            dec = np.load(os.path.join(src, os.path.basename(f)))["dec"]
            ref = lr(wave_of(z["truth"], tfc))
            jobs, keys = [], []
            for sc, e0 in enumerate(z["scen"][:, 0]):
                if esn0 and not np.isclose(esn0, e0).any():
                    continue
                out, g = process(net, tf, z["e0"][sc], z["sd"][sc], z["status"][sc], dev)
                on.setdefault(float(e0), []).append(g.mean())
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
        print(f"\nEs/N0 {e0:g} dB (C/N0 {e0 + CN_OFFSET:g} dB-Hz)、区間 {len(d[methods[0]][2])}、判定がオン {np.mean(on[e0]):.0%}")
        for k in methods:
            s = d[k]
            print(f"  {k:10s} SNR {10 * np.log10(s[0] / s[1]):+6.2f}  PSNR {np.mean(s[3]):6.2f}  ViSQOL {np.mean(s[2]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--awgn", action="store_true")
    ap.add_argument("--esn0", default="", help="--awgn で、この Es/N0 [dB] だけ (カンマ区切り。例: 16,18,20,22)")
    ap.add_argument("--match", default="", help="区間のファイル名にこのどれかを含むものだけ (カンマ区切り。例: Chopin,Scarlatti,Kimiko)")
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load(a.ckpt, dev)
    tf = Transform().to(dev)
    d = os.path.join(a.data, a.profile)
    keys = [k for k in a.match.split(",") if k]
    pick = lambda sub: [f for f in sorted(glob.glob(os.path.join(d, "post", sub, "*.npz")))  # noqa: E731
                        if not keys or any(k in os.path.basename(f) for k in keys)]
    if a.awgn:
        eval_awgn(pick("awgn_val"), os.path.join(d, "mend_awgn", "val"), net, tf, dev,
                  [float(x) for x in a.esn0.split(",") if x])
    else:
        eval_bands(pick("val"), net, tf, dev)


if __name__ == "__main__":
    main()
