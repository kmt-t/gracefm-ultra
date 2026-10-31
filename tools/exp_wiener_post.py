#!/usr/bin/env python3
"""
Haar 合成のあとのポストフィルタの比較 (B200M、AWGN、4 素材 × 2 シード × 複数 Es/N0)。
受信: FloatRx (FEC/CRC は ../gracefm、仮数の復元は float) → 逆 Haar → 後段のフィルタ。欠落はなし。
  raw       処理なし
  fir4k/6k/8k/12k   因果の線形位相 FIR (129 タップ、Kaiser、遅延 64 サンプル = 2 ms) のローパス。評価は遅延を補償して行う
  wiener    周波数領域ウィナー型 (係数の事後分散から漏れる雑音のスペクトルを求める。非因果の STFT、窓 512 点)
SNR と ViSQOL (audio、48 kHz) を、素材と Es/N0 ごとに出す。クリーン受信 (35 dB) での副作用も見る。

  .venv/bin/python tools/exp_wiener_post.py --seconds 5 --esn0 8,12,20,35 --seeds 2
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, firwin, lfilter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm import channel as chan, modem  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm.txrx import Transmitter  # noqa: E402
from gracefm_ultra.plcnet import haar_inv  # noqa: E402
from gracefm_ultra.rx import FloatRx  # noqa: E402
from gracefm_ultra.wiener_post import wiener_postfilter  # noqa: E402
from tools.common import wavio  # noqa: E402

CPUS = {0, 1, 5, 6, 7, 11}
FS = 32000
NTAPS = 129
CUTS = (4000, 6000, 8000, 12000)
MATERIALS = {
    "piano": ("lossless/Emotional_piano.wav", 2.0),
    "percussion": ("lossless/Ceramic_Djembe.wav", 1.0),
    "guitar": ("lossless/344211_giomilko_c-major-9-bossa-nova-guitar.wav", 1.0),
    "speech": ("libri1.ogg", 0.3),
    "demo": ("@demo", 0.0),
    "ambient": ("train_lossless/Raspberrymusic_-_Ambient_(10_minutes).flac", 60.0),
}
FOUR = ["piano", "percussion", "guitar", "speech"]


def load(name, seconds):
    rel, start = MATERIALS[name]
    path = os.path.join(GRACEFM, "eval_audio", rel)
    if rel == "@demo":                                              # GraceFM の合成試験信号 (48 kHz)
        from tools.common.demo_signal import make_demo
        x = make_demo(fs=48000, dur=10.0)
        x = np.stack(x, axis=1) if isinstance(x, tuple) else x
        fs = 48000
        x = wavio.resample(x, fs, FS)
        return x[int(start * FS):int((start + seconds) * FS)] * 0.5
    if path.endswith(".ogg") or path.endswith(".flac"):
        import soundfile as sf
        info = sf.info(path)
        first = int(start * info.samplerate) if rel.startswith("train_lossless") else 0              # 長い曲は途中から使う
        x, fs = sf.read(path, dtype="float64", always_2d=True, start=first, frames=int((seconds + 2) * info.samplerate))
        x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, axis=1)
    else:
        x, fs = wavio.read(path)
    x = wavio.resample(x, fs, FS)
    a0 = 0 if rel.startswith("train_lossless") else int(start * FS)
    return x[a0:a0 + int(seconds * FS)] * 0.5


def job(args):
    name, esn0, seed, seconds, wiener, iirs = args
    os.sched_setaffinity(0, CPUS)
    from visqol import VisqolApi
    api = VisqolApi()
    api.create(mode="audio")
    p = PROFILES["B200M"]
    rx = FloatRx(p)
    lay = rx.lay
    x = load(name, seconds)
    xi = wavio.to_pcm24(x)
    ref = (xi.astype(float) / (1 << 23)).mean(axis=1)
    rng = np.random.default_rng(seed)
    fr = Transmitter(p, 0, 0).run(xi)
    scr = modem.prbs(lay.n_tx)
    n_pre = modem.N_PREAMBLE
    nsym = modem.n_symbols(lay.n_tx, lay.n_strong, p.modulation)
    mean = np.zeros((len(fr), 64))
    var = np.full((len(fr), 64), 1e30)
    bad = 0
    for i, f in enumerate(fr):
        sym = chan.adc(chan.awgn(f / float(1 << modem.SYM_FRAC), esn0, rng))
        llr, _, _ = modem.demodulate(sym[n_pre:n_pre + nsym], lay.n_tx, p.modulation, p.alpha)
        llr = np.where(scr == 1, -llr, llr)
        r = rx.frame(llr)
        if r["ok"]:
            mean[i], var[i] = r["mean"][0], np.minimum(r["var"][0], 1e30)
        else:
            bad += 1
    y0 = np.concatenate([haar_inv(m) for m in mean])
    sc = float(1 << 22)
    n = min(len(y0), len(ref))
    outs = {"raw": y0 / sc}
    for fc in CUTS:
        h = firwin(NTAPS, fc / (FS / 2), window=("kaiser", 8))
        z = lfilter(h, 1.0, np.concatenate([y0 / sc, np.zeros(NTAPS)]))
        outs[f"fir{fc // 1000}k"] = z[(NTAPS - 1) // 2:(NTAPS - 1) // 2 + len(y0)]        # 遅延を補償
    for spec in iirs:                                                              # 因果の IIR (縦続 biquad)。実機と同じ因果の動作
        order, fc = (int(v) for v in spec.split(":"))
        z = sosfilt(butter(order, fc / (FS / 2), output="sos"), y0 / sc)
        lags = range(-2, 10)                                                       # 群遅延ぶんの整数遅れを、相互相関で補償 (SNR 用。ViSQOL は影響が小さい)
        best = max(lags, key=lambda L: float(np.dot(np.roll(z, -L)[2000:-2000], (y0 / sc)[2000:-2000])))
        outs[f"iir{order}_{fc // 1000}k"] = np.roll(z, -best)
    for spec in wiener:
        al, fl = (float(v) for v in spec.split(":"))
        outs["wiener" if spec == "1:0.03" else f"w{al:g}/{fl:g}"] = wiener_postfilter(y0, var, floor=fl, alpha=al)[0] / sc
    valid = np.repeat(var[:, 0] < 1e29, 64)[:n]                                     # 受信できたブロックの係数 (M の 1 成分)
    nbv = len(valid) // 64
    pred = float(np.where(var[:nbv] < 1e29, var[:nbv], 0).sum())                   # 予測した誤差エネルギー (係数領域 = 時間領域)
    actual = float(np.sum(((y0[:n] - ref[:n] * sc) ** 2)[valid]))
    res = {"name": name, "esn0": esn0, "seed": seed, "bad": bad, "calib": actual / max(pred, 1e-30)}
    lo, hi = 1024, n - 1024
    r48 = wavio.resample(ref[lo:hi], FS, 48000)
    for k, z in outs.items():
        zz = z[lo:hi]
        snr = 10 * np.log10(np.sum(ref[lo:hi] ** 2) / max(np.sum((ref[lo:hi] - zz) ** 2), 1e-12))
        mos = float(api.measure_from_arrays(r48, wavio.resample(zz, FS, 48000), sample_rate=48000).moslqo)
        res[k] = (float(snr), mos)
    if name == "piano" and seed == 0:
        out = os.path.join(HERE, "..", "runs", "demo")
        os.makedirs(out, exist_ok=True)
        for k in ("raw", "fir6k", "wiener"):
            wavio.write_pcm24(os.path.join(out, f"post_piano_es{int(esn0)}_{k}.wav"), np.clip(outs[k], -1, 1), FS)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--esn0", default="8,12,20,35")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wiener", default="1:0.03", help="alpha:floor をカンマ区切りで (掃引用)")
    ap.add_argument("--iir", default="", help="order:fc をカンマ区切りで (例 2:6000,4:6000)")
    ap.add_argument("--materials", default=",".join(FOUR), help="カンマ区切り")
    ap.add_argument("--merge", default="", help="カンマ区切りの JSON を結合して表だけ出す (実験は走らせない)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "exp_wiener_post.json"))
    a = ap.parse_args()
    os.sched_setaffinity(0, CPUS)
    if a.merge:
        rows = [r for f in a.merge.split(",") for r in json.load(open(f))]
    else:
        mats = a.materials.split(",")
        jobs = [(m, float(e), s_, a.seconds, a.wiener.split(","), [v for v in a.iir.split(",") if v]) for e in a.esn0.split(",") for m in mats for s_ in range(a.seeds)]
        with Pool(a.workers) as pool:
            rows = []
            for r in pool.imap_unordered(job, jobs):
                rows.append(r)
                print(f"{r['name']:10} Es/N0 {r['esn0']:4.0f} seed {r['seed']} CRC失敗 {r['bad']}", flush=True)
        json.dump(rows, open(a.out, "w"))
    names = list(dict.fromkeys(r["name"] for r in rows))
    keys = [k for k in rows[0] if k not in ("name", "esn0", "seed", "bad", "calib")]
    print(f"事後分散の較正: 実際の誤差エネルギー / 予測 (1 が理想) = {np.exp(np.mean(np.log([max(r['calib'], 1e-9) for r in rows]))):.2f} (幾何平均)")
    print(f"\n{len(names)} 素材 ({', '.join(names)}) × {len({(r['name'], r['seed']) for r in rows}) // len(names)} シード。SNR [dB] / ViSQOL の平均 (FIR は 129 タップ、遅延 {(NTAPS - 1) // 2} サンプル = {(NTAPS - 1) // 2 / FS * 1000:.1f} ms を補償して評価)")
    print(f"{'Es/N0':>6}" + "".join(f"{k:>15}" for k in keys))
    for e in sorted({r["esn0"] for r in rows}):
        sel = [r for r in rows if r["esn0"] == e]
        print(f"{e:6.0f}" + "".join(f"{np.mean([r[k][0] for r in sel]):7.2f}/{np.mean([r[k][1] for r in sel]):6.3f} " for k in keys))
    print("\n素材別 (Es/N0 = 最大のクリーン受信と 12 dB)")
    for e in (max(r["esn0"] for r in rows), 12.0):
        for m in names:
            sel = [r for r in rows if r["esn0"] == e and r["name"] == m]
            if sel:
                print(f"{m:10}{e:4.0f}" + "".join(f"{np.mean([r[k][0] for r in sel]):7.2f}/{np.mean([r[k][1] for r in sel]):6.3f} " for k in keys))


if __name__ == "__main__":
    main()
