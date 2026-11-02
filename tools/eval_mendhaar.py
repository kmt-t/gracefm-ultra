#!/usr/bin/env python3
"""
MendHaarNet の評価。検証データの同じ窓で、使えないフレームの SNR [dB] を、使えない連続区間の長さごとに、
AR 補間・旧 MendNet (係数を出す版、--old)・MendHaarNet の前段だけ (LLR → E0)・MendHaarNet で比べる。使えるフレームの SNR と、
怪しさ (後段の log 分散) が実際の誤差と合っているか (フレームごとの順位相関) も出す。

  .venv/bin/python tools/eval_mendhaar.py --ckpt runs/mendhaar_B200.pt --old runs/mend_B200_v2r.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_mend import BUCKETS, ar_fill  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.haarnet import make_meta  # noqa: E402
from gracefm_ultra.mendhaar import HaarStage, estimate  # noqa: E402
from gracefm_ultra.mendnet import MendNet, build_input, compose  # noqa: E402
from train_mend import CPUS, Windows  # noqa: E402
from train_mendhaar import load_front, step_batch  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mend", default="", help="前段のチェックポイント (なければ --ckpt に記録した場所)")
    ap.add_argument("--old", default="", help="旧 MendNet (係数を出す版) のチェックポイント")
    ap.add_argument("--windows", type=int, default=1500)
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(len(CPUS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = os.path.join(a.data, a.profile, "mend")
    files = sorted(glob.glob(os.path.join(d, "val", "*.npz")))
    va = Windows(files, dev, bits=True)
    z = np.load(files[0])
    meta = make_meta({k: z[k] for k in ("pos", "coef_of_bit", "group")}, PROFILES[a.profile]).to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    front, _ = load_front(a.mend or ck["mend"], dev)
    net = HaarStage(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    old = None
    if a.old:
        oc = torch.load(a.old, map_location=dev)
        old = MendNet(**oc["cfg"]).to(dev).eval()
        old.load_state_dict(oc["model"])
    rng = np.random.default_rng(7)
    methods = ["AR", "旧MendNet", "前段だけ", "MendHaarNet"]
    acc, keep = {}, {m: [0.0, 0.0] for m in ["復号のまま", "旧MendNet", "前段だけ", "MendHaarNet"]}
    unc, err = [], []                                                    # フレームごとの怪しさ (log 分散の平均)、誤差 (S で割った二乗誤差)
    for _ in range(a.windows):
        locs = va.locate(1, rng)
        llr, st, status, truth, _ = step_batch(front, va, locs, meta.nbit)
        with torch.no_grad():
            est, S, e0, logvar = estimate(net, llr, st, status, meta)
        per = [va.gather(locs, [ch]) for ch in range(2)]
        t = truth[0].cpu().numpy() / 2 ** 22
        us = status[0].cpu().numpy() <= 1
        run = runs_of(us)
        outs = {"前段だけ": e0[0].cpu().numpy() / 2 ** 22, "MendHaarNet": est[0].cpu().numpy() / 2 ** 22}
        c = np.concatenate([per[0]["coef"][0].cpu().numpy(), per[1]["coef"][0].cpu().numpy()], -1)
        outs["AR"] = np.concatenate([ar_fill(c[:, :64], us), ar_fill(c[:, 64:], us)], -1)
        if old is not None:
            parts = []
            for b in per:
                with torch.no_grad():
                    x, sig, usable = build_input(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], b["hint_h"], b["hint_q"])
                    parts.append(compose(old(x), b["coef"], sig, usable, oc.get("refine", False))[0].cpu().numpy())
            outs["旧MendNet"] = np.concatenate(parts, -1)
        for k in ("復号のまま", "旧MendNet", "前段だけ", "MendHaarNet"):
            y = c if k == "復号のまま" else outs.get(k)
            if y is not None:
                keep[k][0] += float((t[us] ** 2).sum())
                keep[k][1] += float(((y[us] - t[us]) ** 2).sum())
        for name, lo, hi in BUCKETS:
            m = (~us) & (run >= lo) & (run <= hi)
            if not m.any():
                continue
            for k in methods:
                if k in outs:
                    s = acc.setdefault((name, k), [0.0, 0.0])
                    s[0] += float((t[m] ** 2).sum())
                    s[1] += float(((outs[k][m] - t[m]) ** 2).sum())
        e2 = (((est[0] / S[0]) - truth[0] / S[0]) ** 2).mean(1)
        unc.append(logvar[0].mean(1).cpu().numpy())
        err.append(np.log(e2.cpu().numpy() + 1e-9))
    for name, _, _ in BUCKETS:
        print(f"{name:10s}", "  ".join(f"{k} {10 * np.log10(acc[(name, k)][0] / acc[(name, k)][1]):+6.2f} dB" for k in methods if (name, k) in acc))
    print("使えるフレーム", "  ".join(f"{k} {10 * np.log10(v[0] / v[1]):+6.2f} dB" for k, v in keep.items() if v[1] > 0))
    print(f"怪しさと実際の誤差の順位相関 {rank_corr(np.concatenate(unc), np.concatenate(err)):.3f}")


if __name__ == "__main__":
    main()
