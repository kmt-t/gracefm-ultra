#!/usr/bin/env python3
"""
HaarNet 用 (--mend なら MendNet 用) の学習データを作る (Rust の engine/gfm-gen を使う)。
音源 1 本から複数の区間 (seconds + 2 秒の枠ごとに 1 区間) を取り、符号化 → 通信路シナリオ → 受信 LLR を data/<profile>/<sub>/<split>/ に書く。
train / val は音源の番号で分ける (7 本に 1 本が val)。--awgn を与えると、通信路はランダムではなく、その Es/N0 ごとの AWGN のみ (フェージング・人工の欠落なし)。出力は data/<profile>/<sub>/<split>/*.npz (受信 LLR、送信ビット、刻み、量子化前後の係数、シナリオ)。

  .venv/bin/python tools/gen_hn.py --profile B200 --split train --segments 4 --workers 2 --sub hn_big
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from tools.common import wavio  # noqa: E402

BIN = os.path.join(HERE, "..", "engine", "target", "release", "gfm-gen")


def scenarios(rng, n, esn0_min=0.0):
    """多様な通信路のリスト [(基準Es/N0, フェージング, 欠落率, 平均バースト長)]"""
    out = []
    for i in range(n):
        esn0 = float(rng.uniform(esn0_min, 16)) if rng.random() < 0.5 else float(rng.uniform(16, 40))      # 高 C/N (同軸直結など) も厚く
        fade = int(rng.random() < 0.5)
        rate = 0.0 if rng.random() < 0.5 else float(rng.choice([0.02, 0.05, 0.1, 0.2, 0.3]))
        burst = float(rng.choice([1, 3, 8, 25]))
        out.append((esn0, fade, rate, burst))
    return out


def read_segment(path, start_s, dur_s):
    """音源の start_s 秒から dur_s 秒を float の (n, 2) で読む。→ (x, fs)"""
    if path.endswith(".wav"):
        x, fs = wavio.read(path)
        a = int(start_s * fs)
        return x[a:a + int(dur_s * fs)], fs
    import soundfile as sf
    info = sf.info(path)
    x, fs = sf.read(path, dtype="float64", always_2d=True, start=int(start_s * info.samplerate), frames=int(dur_s * info.samplerate))
    x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, axis=1)
    return x, fs


def job(args):
    path, slot, profile, seconds, scen, seed, outdir, mend = args
    pin()
    p = PROFILES[profile]
    name = os.path.splitext(os.path.basename(path))[0][:50]
    out = os.path.join(outdir, f"{name}_{slot}_{seed}.npz")
    if os.path.exists(out):
        return out, 0, 0, 0.0
    t0 = time.time()
    span = seconds + 2
    x, fs = read_segment(path, slot * span, span)
    xs = wavio.resample(x, fs, p.fs)
    a = int(0.5 * p.fs)                                    # 先頭の再標本化の端を避ける
    n = int(seconds * p.fs)
    pcm = wavio.to_pcm24(xs[a:a + n] * 0.5)
    nfr = len(pcm) // p.block
    pcm = pcm[:nfr * p.block]
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "x.raw")
        pcm.astype("<i4").tofile(raw)
        cmd = [BIN, "--profile", profile, "--pcm", raw, "--seed", str(seed), "--out", out + ".tmp", "--threads", "1", "--start", str(int(slot * span * 1000))]
        if mend:
            cmd.append("--mend")
        for e0, fade, rate, burst in scen:
            cmd += ["--scen", f"{e0},{fade},{rate},{burst}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-500:])
    os.replace(out + ".tmp", out)
    return out, nfr, len(scen), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--src", default=os.path.join(GRACEFM, "eval_audio", "train_lossless"))
    ap.add_argument("--segments", type=int, default=4, help="音源 1 本あたりの区間数の上限")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub", default="hn_big", help="出力先 data/<profile>/<sub>/<split>")
    ap.add_argument("--esn0-min", type=float, default=0.0, help="シナリオの基準 Es/N0 の下限 [dB]")
    ap.add_argument("--awgn", default="", help="AWGN のみの Es/N0 [dB] の並び (例: 16,18,20)。与えるとランダムなシナリオの代わりに使う")
    ap.add_argument("--mend", action="store_true", help="MendNet 用 (復号した係数・状態・分散) を書く")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    a = ap.parse_args()
    pin()
    files = sorted(f for f in os.listdir(a.src) if f.endswith((".flac", ".wav")))
    outdir = os.path.join(a.data, a.profile, a.sub, a.split)
    os.makedirs(outdir, exist_ok=True)
    import soundfile as sf
    jobs = []
    for i, f in enumerate(files):
        if (i % 7 == 3) != (a.split == "val"):
            continue
        path = os.path.join(a.src, f)
        dur = wavio.read(path)[0].shape[0] / wavio.read(path)[1] if f.endswith(".wav") else sf.info(path).duration
        nslots = max(int(dur // (a.seconds + 2)), 1)
        rng = np.random.default_rng(a.seed * 1000 + i)
        for slot in rng.permutation(nslots)[:a.segments]:
            seed = a.seed * 100000 + i * 100 + int(slot)
            scen = [(float(e), 0, 0.0, 1.0) for e in a.awgn.split(",")] if a.awgn else scenarios(np.random.default_rng(seed), a.scenarios, a.esn0_min)
            jobs.append((path, int(slot), a.profile, a.seconds, scen, seed, outdir, a.mend))
    print(f"{a.profile} {a.split}: {len(jobs)} 区間 → {outdir}", flush=True)
    t0 = time.time()
    with Pool(min(a.workers, len(jobs), len(CPUS))) as pool:
        for k, (out, nfr, S, dt) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{k}/{len(jobs)}] {os.path.basename(out)}: {nfr} frames x {S} scenarios, {dt:.0f}s (経過 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
