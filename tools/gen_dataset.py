#!/usr/bin/env python3
"""
学習データの生成。音源を GraceFM で符号化し、複数の通信路シナリオ (Es/N0、フェージング、フレーム欠落) を通して、
フレームごとに (正解ビット, 受信LLR, 正解係数) を data/<profile>/<clip>.npz に保存する。
符号化は音源ごとに一度だけで、通信路だけシナリオごとに変える。FEC復号はしない (学習側が行う)。

  .venv/bin/python tools/gen_dataset.py --profile B200M --clips 4 --seconds 10 --workers 6

保存するもの (シナリオ s ごとに、フレーム数 F、送信ビット数 N):
  bits   (F, N)    uint8   送信ビット (スクランブル前の framing.pack の出力)
  llr    (s, F, N) int8    受信LLR (スクランブルを戻した後、LLR_FRAC=2 の整数。正なら bit 0 寄り。欠落フレームは 0)
  var    (s, F)    int32   軸あたりの雑音分散の推定 (欠落は 0)
  lost   (s, F)    bool    欠落フレーム
  esn0   (s, F)    float32 フレームごとの Es/N0 [dB] (フェージング込み)
  coef   (F, 2, 64) int32  量子化後の正解Haar係数 (M, S)。帯域 k=0..4 を連結
  coef0  (F, 2, 64) int32  量子化前のHaar係数 (M, S)
  scen   (s, 4)    float32 [基準 Es/N0, フェージング(0:awgn 1:rayleigh), 欠落率, 平均バースト長]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))

from gracefm import channel as chan, framing, modem  # noqa: E402
from gracefm.codec import encode_block, haar_forward  # noqa: E402
from gracefm.decoder import reconstruct_coeffs  # noqa: E402
from gracefm.layout import Layout  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402
from gracefm_ultra.channels import loss_mask  # noqa: E402

CPUS = {int(v) for v in os.environ.get("ULTRA_CPUS", "0,1,5,6,7,11").split(",")}      # 使ってよいコア (環境変数で変更)


def scenarios(rng, n):
    """多様な通信路のリスト [(基準Es/N0, フェージング, 欠落率, 平均バースト長)]"""
    out = []
    for i in range(n):
        esn0 = float(rng.uniform(0, 16)) if rng.random() < 0.5 else float(rng.uniform(16, 40))      # 高 C/N (同軸直結など) も厚く
        fade = int(rng.random() < 0.5)
        rate = 0.0 if rng.random() < 0.5 else float(rng.choice([0.02, 0.05, 0.1, 0.2, 0.3]))
        burst = float(rng.choice([1, 3, 8, 25]))
        out.append((esn0, fade, rate, burst))
    return out


def make_clip(args):
    path, profile_name, seconds, nscen, seed, outdir = args
    os.sched_setaffinity(0, CPUS)
    p = PROFILES[profile_name]
    lay = Layout(p)
    name = os.path.splitext(os.path.basename(path))[0][:60]
    rng = np.random.default_rng(seed)
    x, fs = wavio.read(path)
    n = int(seconds * p.fs)
    xs = wavio.resample(x[: int((seconds + 2) * fs * 1.0) + 1 if len(x) > 3 * fs * seconds else len(x)], fs, p.fs)
    start = int(rng.integers(0, max(1, len(xs) - n))) if len(xs) > n else 0
    xs = wavio.to_pcm24(xs[start:start + n] * 0.5)
    b = p.block
    nfr = len(xs) // b
    t0 = time.time()

    bits = np.zeros((nfr, lay.n_tx), np.uint8)
    sym = []
    coef = np.zeros((nfr, 2, b), np.int32)
    coef0 = np.zeros((nfr, 2, b), np.int32)
    for i in range(nfr):
        blk = xs[i * b:(i + 1) * b]
        bd = encode_block(blk, p)
        bt = framing.pack(lay, bd, 0, i, None)
        bits[i] = bt
        buf = np.zeros(modem.n_symbols(lay.n_tx, lay.n_strong, p.modulation) * modem.bits_per_symbol(p.modulation), np.int64)
        buf[: len(bt)] = bt
        buf ^= modem.prbs(len(buf))
        sym.append(modem.modulate(buf, len(buf) // modem.bits_per_symbol(p.modulation), p.modulation, p.alpha))
        co = reconstruct_coeffs(bd, p)
        ms = {"M": (blk[:, 0] + blk[:, 1] + 2) >> 2, "S": (blk[:, 0] - blk[:, 1] + 2) >> 2}
        for ci, ch in enumerate(("M", "S")):
            coef[i, ci] = np.concatenate(co[ch])
            coef0[i, ci] = np.concatenate(haar_forward(ms[ch], p.levels))
    scr = modem.prbs(lay.n_tx)

    scen = scenarios(rng, nscen)
    S = len(scen)
    llr = np.zeros((S, nfr, lay.n_tx), np.int8)
    var = np.zeros((S, nfr), np.int64)
    lost = np.zeros((S, nfr), bool)
    esn0 = np.zeros((S, nfr), np.float32)
    for si, (e0, fade, rate, burst) in enumerate(scen):
        g = chan.fading_gains(nfr, "rayleigh" if fade else "awgn", np.random.default_rng(seed * 1000 + si))
        lost[si] = loss_mask(nfr, rate, burst, np.random.default_rng(seed * 1000 + 500 + si))
        crng = np.random.default_rng(seed * 1000 + 900 + si)
        for i in range(nfr):
            e = e0 + 10 * np.log10(max(g[i], 1e-6))
            esn0[si, i] = e
            if lost[si, i]:
                continue
            rx = chan.adc(chan.awgn(sym[i] / float(1 << modem.SYM_FRAC), e, crng))
            l, h, v = modem.demodulate(rx[modem.N_PREAMBLE:], lay.n_tx, p.modulation, p.alpha)
            llr[si, i] = np.clip(np.where(scr == 1, -l, l), -127, 127)
            var[si, i] = v
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}_{seed}.npz")
    np.savez(out, bits=bits, llr=llr, var=var, lost=lost, esn0=esn0, coef=coef, coef0=coef0,
             scen=np.array(scen, np.float32), profile=profile_name, source=os.path.basename(path), start=start)
    return out, nfr, S, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200M", choices=PROFILES)
    ap.add_argument("--src", default=os.path.join(GRACEFM, "eval_audio", "train_lossless"))
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--scenarios", type=int, default=6, help="音源あたりの通信路シナリオ数")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val"], default="train", help="val は 7 本に 1 本 (固定)。train はそれ以外")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    files = sorted(f for f in os.listdir(a.src) if f.endswith((".flac", ".wav")))
    files = [f for i, f in enumerate(files) if (i % 7 == 3) == (a.split == "val")]
    rng = np.random.default_rng(a.seed)
    pick = rng.permutation(len(files))[:a.clips]
    outdir = os.path.join(a.out, a.profile, a.split)
    jobs = [(os.path.join(a.src, files[j]), a.profile, a.seconds, a.scenarios, a.seed * 100000 + int(j), outdir) for j in pick]
    with Pool(min(a.workers, len(jobs), len(CPUS))) as pool:
        for out, nfr, S, dt in pool.imap_unordered(make_clip, jobs):
            print(f"{os.path.basename(out)}: {nfr} frames x {S} scenarios, {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
