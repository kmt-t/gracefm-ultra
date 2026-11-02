#!/usr/bin/env python3
"""
HaarNet-lite の学習。gfm-gen（tools/gen_hn.py）が作るデータから、64 フレームの窓をランダムに切って、
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
sys.path.insert(0, HERE)
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.haarnet import WIN, HaarNetLite, features, haar_matrix, make_meta  # noqa: E402


def load(split, profile, sub="hn_big"):
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "data", profile, sub, split, "*.npz"))):
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


_HM = {}


_CIDX = {}


def set_cidx(cidx, dev):
    """coef_offset 順の係数番号 → (成分, 64 係数) の平坦な添字。副系 (k=0..2 だけ) は高域の係数を 0 で埋めるために使う"""
    _CIDX["c"] = torch.from_numpy(np.asarray(cidx, np.int64))


def _waveform(coef, ch_of_coef=None):
    """coef: (B, T, ncoef) → 各成分 (M, S) の時間波形 (B, nch, T*64)。係数は、ない帯域を 0 で埋めて、成分あたり 64 係数にして逆 Haar にかける"""
    key = coef.device
    if key not in _HM:
        _HM[key] = torch.from_numpy(haar_matrix()).float().to(key)
    cidx = _CIDX["c"].to(key)
    full = torch.zeros(coef.shape[0], coef.shape[1], 128, device=key, dtype=coef.dtype)
    full[..., cidx] = coef
    ncomp = 2 if int(cidx.max()) >= 64 else 1
    outs = []
    for ci in range(ncomp):
        x = full[..., ci * 64:(ci + 1) * 64] @ _HM[key]                           # (B, T, 64)
        outs.append(x.reshape(x.shape[0], -1))
    return torch.stack(outs, 1)


def spectral_loss(est, t, S, keep, nfft=512, hop=128, fmin_bin=32, floor=1e-3):
    """再構成波形の対数振幅 (2 kHz 以上 = 512 点 FFT で 32 ビン以上) の L1。漏れ雑音 (高域の広帯域雑音) を直接抑える。
    est, t: 正規化した係数 (B, T, ncoef)、S: (B, 1, ncoef) 尺度、keep: (B, T, 1)"""
    w = _waveform(est * S, None) / S.mean()
    r = _waveform(t * S, None) / S.mean()
    B, C, N = w.shape
    win = torch.hann_window(nfft, device=w.device)
    Xw = torch.stft(w.reshape(B * C, N), nfft, hop, window=win, return_complex=True).abs()
    Xr = torch.stft(r.reshape(B * C, N), nfft, hop, window=win, return_complex=True).abs()
    lo = torch.log(Xw[:, fmin_bin:] + floor)
    ro = torch.log(Xr[:, fmin_bin:] + floor)
    nf = Xw.shape[-1]
    fr = torch.arange(nf, device=w.device) * hop + nfft // 2                       # 各 STFT フレームの中心サンプル
    kf = keep[..., 0].repeat_interleave(64, dim=1)[:, :N]                         # (B, N) サンプル単位の有効
    kf = kf.repeat_interleave(C, dim=0)
    use = kf[:, fr.clamp(max=N - 1)]                                              # (B*C, nf)
    return ((lo - ro).abs().mean(1) * use).sum() / use.sum().clamp_min(1.0)


def loss_fn(net, meta, llr, step, var, lost, truth, edge=4, lam_spec=0.05, lam_harm=0.5, spec_floor=1e-3):
    x, S, e0n = features(llr, step, var, lost, meta)
    est = e0n + net(x)
    t = truth / S
    keep = (~lost).float()[..., None].clone()
    keep[:, :edge] = 0
    keep[:, -edge:] = 0
    w = S ** 2
    nvalid = keep[..., 0].sum(1)
    err_c = ((est - t) ** 2) * w * keep                                                # (B, T, ncoef)
    err0_c = ((e0n - t) ** 2) * w * keep
    ref = ((t ** 2) * w * keep).sum((1, 2)) + 0.05 * nvalid * w.sum((1, 2)) + 1e-6      # 下限: 観測レベルの -13 dB
    err, err0 = err_c.sum((1, 2)), err0_c.sum((1, 2))
    # 成分 (M, S) ごとに相対誤差を出して平均する (S の誤差も同じ重みで効かせる)
    half = meta.ncoef // 2 if meta.ncoef >= 128 else meta.ncoef
    if meta.ncoef >= 128:
        m_rel = err_c[..., :half].sum((1, 2)) / (((t[..., :half] ** 2) * w[..., :half] * keep).sum((1, 2)) + 0.05 * nvalid * w[..., :half].sum((1, 2)) + 1e-6)
        s_rel = err_c[..., half:].sum((1, 2)) / (((t[..., half:] ** 2) * w[..., half:] * keep).sum((1, 2)) + 0.05 * nvalid * w[..., half:].sum((1, 2)) + 1e-6)
        base = 0.5 * (m_rel + s_rel).mean()
    else:
        base = (err / ref).mean()
    harm = (torch.relu(err - err0) / ref).mean()                                       # E0 より悪くした分に罰則
    spec = spectral_loss(est, t, S, keep, floor=spec_floor)
    return base + lam_harm * harm + lam_spec * spec, (err0 / ref).mean(), (err / ref).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ch", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--train-sub", default="hn_big", help="学習データの置き場 data/<profile>/<train-sub>/train")
    ap.add_argument("--cpu", action="store_true", help="GPU があっても CPU で学習する")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="", help="このチェックポイントの重みから始める")
    ap.add_argument("--lam-spec", type=float, default=0.05, help="時間領域スペクトル損失 (2 kHz 以上の対数振幅) の重み")
    ap.add_argument("--spec-floor", type=float, default=1e-3, help="その対数の下限 (正規化単位)。静かな高域を効かせるには 1e-6 程度")
    ap.add_argument("--lam-harm", type=float, default=0.5, help="E0 より悪化した分への罰則の重み")
    ap.add_argument("--min-esn0", type=float, default=-1e9, help="この Es/N0 [dB] 以上のシナリオだけで学習する")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "runs", f"hn_{a.profile}.pt")
    pin()
    torch.set_num_threads(min(len(CPUS), 4))
    rng = np.random.default_rng(0)
    data = load("train", a.profile, a.train_sub)
    meta = make_meta(data[0], PROFILES[a.profile])
    from hn_index import dataset_index
    from gracefm.layout import Layout
    _cidx, _ = dataset_index(PROFILES[a.profile], Layout(PROFILES[a.profile]))
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    cfg = dict(nbit=meta.nbit, ncoef=meta.ncoef, ch=a.ch, blocks=a.blocks)
    net = HaarNetLite(**cfg).to(dev)
    meta.to(dev)
    set_cidx(_cidx, dev)
    print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""), flush=True)
    print(f"{a.profile}: 学習 {len(data)} 音源、bit {meta.nbit}、係数 {meta.ncoef}、パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev)["model"])
        print(f"{a.resume} から再開 (Es/N0 >= {a.min_esn0})", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    t0, step, run, run0 = time.time(), 0, [], []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        loss, base, rel = loss_fn(net, meta, *[b.to(dev, non_blocking=True) for b in batch(data, a.bs, rng, a.min_esn0)], lam_spec=a.lam_spec, lam_harm=a.lam_harm, spec_floor=a.spec_floor)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(rel.item()); run0.append(base.item())
        step += 1
        if step % 50 == 0:
            print(f"step {step:5d} {time.time() - t0:5.0f}s  E0 {-10 * np.log10(np.mean(run0[-50:])):6.2f} dB  net {-10 * np.log10(np.mean(run[-50:])):6.2f} dB", flush=True)
        if step % 500 == 0:
            torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, "cfg": cfg, "step": step}, out)
    torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, "cfg": cfg, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
