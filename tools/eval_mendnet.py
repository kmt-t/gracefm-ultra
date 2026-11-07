#!/usr/bin/env python3
"""
MendNet の評価。MendNet の係数は、出力の LLR から作る仮数の期待値 × 刻み (E0、mendnet.soft_e0)。怪しさは、その事後標準偏差。

- 既定: 検証データ (data/<profile>/mend/val) の窓で、使えないフレームの SNR [dB] を使えない連続区間の長さごとに、使えるフレームの SNR、
  怪しさと実際の誤差の順位相関 (フレームごと) を出す。
- --awgn: AWGN のみの検証データ (gen_mend.py --split val --sub mend_awgn --awgn 16,18,…,30) の各シナリオを丸ごと (約 20 秒) 復元し、
  音声 (ステレオ 32 kHz) で原音と比べる。窓 WIN フレームを WIN/2 ずつずらし、各窓の中央 WIN/2 フレームを採る (実際の受信と同じ)。
  指標は SNR (全体)、PSNR (満点 ±1、区間ごとの平均)、ViSQOL (audio モード、48 kHz に変換、区間ごとの平均) を Es/N0 ごとに。
  C/N0 [dB-Hz] は Es/N0 + 50 (B200、B200-L の換算)。

  .venv/bin/python tools/eval_mendnet.py --ckpt runs/mendnet_B200-L.pt
  .venv/bin/python tools/eval_mendnet.py --ckpt runs/mendnet_B200-L.pt --awgn
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
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.transform import coef_power, inverse  # noqa: E402
from gracefm_ultra.mendnet import Q22, WIN, MendNet, make_meta, run_both, soft_e0  # noqa: E402
from tools.common import wavio  # noqa: E402
from mend_windows import Windows  # noqa: E402

HOP = WIN // 2
BUCKETS = [("連続1", 1, 1), ("連続2-3", 2, 3), ("連続4-8", 4, 8), ("連続9-25", 9, 25), ("連続26以上", 26, 10**9)]
METHODS = ["復号のまま", "MendNet"]


def cn_offset(p):
    """C/N0 [dB-Hz] − Es/N0 [dB]: シンボルレートの dB"""
    from gracefm.encoder import Encoder
    from gracefm.modem import N_PREAMBLE
    return 10 * np.log10((N_PREAMBLE + Encoder(p).nsym) * p.fs / p.block)


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    net = MendNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net


def coefs(win, locs, net, meta):
    """窓 locs の係数 (B,T,128) 満点 1.0: {"復号のまま", "MendNet"} と、MendNet の事後標準偏差 (B,T,128)、状態 (B,T)、原音"""
    per = [win.gather(locs, [ch] * len(locs)) for ch in range(2)]
    with torch.no_grad():
        llr, step = run_both(net, per, win.bit_idx.to(win.device), win.bit_ok.to(win.device), meta.nbit)
        e0, sd = soft_e0(llr, step, meta, torch.cat([b["coef"] for b in per], 2), per[0]["status"].long())
    cat = lambda k: torch.cat([per[0][k], per[1][k]], -1)  # noqa: E731
    out = {"復号のまま": cat("coef"), "MendNet": e0 / Q22}
    return {k: v.cpu().numpy() for k, v in out.items()}, (sd / Q22).cpu().numpy(), per[0]["status"].cpu().numpy(), cat("target").cpu().numpy()


def runs_of(us):
    """各フレームが属する、使えない連続区間の長さ (使えるフレームは 0)"""
    run = np.zeros(len(us), int)
    i = 0
    while i < len(us):
        if us[i]:
            i += 1
            continue
        j = i
        while j < len(us) and not us[j]:
            j += 1
        run[i:j] = j - i
        i = j
    return run





def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def eval_windows(va, net, meta, n, kind):
    methods = ["MendNet"]
    acc, keep = {}, {m: [0.0, 0.0] for m in METHODS}
    unc, err = [], []
    rng = np.random.default_rng(7)
    w = np.sqrt(np.tile(coef_power(kind), 2)).astype(np.float32)                    # 誤差を波形でのエネルギーで数える
    for _ in range(n):
        out, sd, status, t = coefs(va, va.locate(1, rng), net, meta)
        out, sd, t = {k: v[0] * w for k, v in out.items()}, sd[0] * w, t[0] * w
        us = status[0] <= 1
        run = runs_of(us)
        for k in keep:
            keep[k][0] += float((t[us] ** 2).sum())
            keep[k][1] += float(((out[k][us] - t[us]) ** 2).sum())
        for name, lo, hi in BUCKETS:
            m = (~us) & (run >= lo) & (run <= hi)
            if m.any():
                for k in methods:
                    s = acc.setdefault((name, k), [0.0, 0.0])
                    s[0] += float((t[m] ** 2).sum())
                    s[1] += float(((out[k][m] - t[m]) ** 2).sum())
        unc.append(np.log((sd ** 2).mean(1) + 1e-20))
        err.append(np.log(((out["MendNet"] - t) ** 2).mean(1) + 1e-20))
    for name, _, _ in BUCKETS:
        if (name, "MendNet") in acc:
            print(f"{name:10s}", "  ".join(f"{k} {10 * np.log10(acc[(name, k)][0] / acc[(name, k)][1]):+6.2f} dB" for k in methods))
    print("使えるフレーム", "  ".join(f"{k} {10 * np.log10(v[0] / v[1]):+6.2f} dB" for k, v in keep.items()))
    print(f"怪しさ (事後標準偏差) と実際の誤差の順位相関 {rank_corr(np.concatenate(unc), np.concatenate(err)):.3f}")


_api = None


def visqol(args):
    """(原音, 出力, 標本化周波数) (N, 2) → ViSQOL (両チャンネルの平均)。ワーカーごとに API を作る"""
    global _api
    if _api is None:
        from visqol import VisqolApi
        _api = VisqolApi()
        _api.create(mode="audio")
    ref, y, fs = args
    return float(np.mean([_api.measure_from_arrays(wavio.resample(ref[:, c], fs, 48000), wavio.resample(y[:, c], fs, 48000), sample_rate=48000).moslqo
                          for c in (0, 1)]))


def to_audio(c, kind):
    """係数 (N,128) (M 64、S 64) → L/R (N×64, 2)"""
    m, s = inverse(c[:, :64], kind), inverse(c[:, 64:], kind)
    return np.stack([m + s, m - s], 1)


def stream(va, si, sc, net, meta, chunk=32):
    """シナリオ (si, sc) を窓でつないで復元する。→ ({方法: 係数 (N,128)}、原音 (N,128))"""
    n = va.truth[si].shape[0]
    starts = list(range(0, n - WIN + 1, HOP))
    a0, a1 = starts[0] + WIN // 4, starts[-1] + WIN // 4 + HOP
    res, ref = {}, np.zeros((a1 - a0, 128), np.float32)
    for i in range(0, len(starts), chunk):
        locs = [(si, sc, a) for a in starts[i:i + chunk]]
        out, _, _, t = coefs(va, locs, net, meta)
        for j, (_, _, a) in enumerate(locs):
            lo, mid = a + WIN // 4 - a0, slice(WIN // 4, WIN // 4 + HOP)
            for k, v in out.items():
                res.setdefault(k, np.zeros((a1 - a0, 128), np.float32))[lo:lo + HOP] = v[j, mid]
            ref[lo:lo + HOP] = t[j, mid]
    return res, ref


def eval_awgn(va, files, net, meta, p):
    kind = p.transform
    methods = METHODS
    acc = {}
    with Pool(len(CPUS), initializer=pin) as pool:
        for si, f in enumerate(files):
            jobs, keys = [], []
            for sc, e0 in enumerate(np.load(f)["scen"][:, 0]):
                out, t = stream(va, si, sc, net, meta)
                ref = to_audio(t, kind)
                for k in methods:
                    y = to_audio(out[k], kind)
                    e = float(((y - ref) ** 2).sum())
                    s = acc.setdefault(float(e0), {}).setdefault(k, [0.0, 0.0, [], []])
                    s[0] += float((ref ** 2).sum())
                    s[1] += e
                    s[3].append(10 * np.log10(ref.size / max(e, 1e-20)))
                    jobs.append((ref, y, p.fs))
                    keys.append((float(e0), k))
            for (e0, k), mos in zip(keys, pool.map(visqol, jobs)):
                acc[e0][k][2].append(mos)
            print(f"[{si + 1}/{len(files)}] {os.path.basename(f)}", flush=True)
    print("\nSNR [dB] (全体)、PSNR [dB] (区間の平均)、ViSQOL (区間の平均)")
    for e0, d in sorted(acc.items()):
        print(f"\nEs/N0 {e0:g} dB (C/N0 {e0 + cn_offset(p):.1f} dB-Hz)、区間 {len(d[methods[0]][2])}")
        for k in methods:
            s = d[k]
            print(f"  {k:10s} SNR {10 * np.log10(s[0] / s[1]):+6.2f}  PSNR {np.mean(s[3]):6.2f}  ViSQOL {np.mean(s[2]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200-L")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True, help="MendNet のチェックポイント")
    ap.add_argument("--awgn", action="store_true", help="AWGN のみの検証データ (data/<profile>/mend_awgn/val) を音声で評価する")
    ap.add_argument("--windows", type=int, default=1500)
    a = ap.parse_args()
    pin()
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(a.data, a.profile, "mend_awgn" if a.awgn else "mend", "val", "*.npz")))
    va = Windows(files, dev)
    z = np.load(files[0])
    meta = make_meta({k: z[k] for k in ("pos", "coef_of_bit")}, PROFILES[a.profile]).to(dev)
    net = load(a.ckpt, dev)
    if a.awgn:
        eval_awgn(va, files, net, meta, PROFILES[a.profile])
    else:
        eval_windows(va, net, meta, a.windows, PROFILES[a.profile].transform)


if __name__ == "__main__":
    main()
