#!/usr/bin/env python3
"""
B200 (ステレオ) のサンプル音声。AWGN の通信路を通して、次の各出力を WAV (32 kHz、24 bit、ステレオ) に書き出し、SNR と ViSQOL を出す。
  gfm         現行 GraceFM 受信機 (FEC/CRC、LPC 補間あり)
  e0          生LLR の事後平均 (FEC なし、指数は復号済みとする)
  ultra       HaarNet-lite (M・S とも)
  ultra_M     M 成分だけ HaarNet-lite、S 成分は e0
指数 (刻み) は、符号化時の値を使う (試作の仮定)。欠落フレームはなし。

  .venv/bin/python tools/demo_b200.py --seconds 8 --esn0 8,12,16,24
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm import channel as chan, modem  # noqa: E402
from gracefm.layout import Layout  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm.txrx import Transmitter, receive  # noqa: E402
from gracefm_ultra.haarnet import WIN, HaarNetLite, features, make_meta, soft_e0  # noqa: E402
from gracefm_ultra.plcnet import haar_inv  # noqa: E402
from prep_haarnet import steps_of  # noqa: E402
from tools.common import wavio  # noqa: E402


def to_lr(coef, nfr):
    """coef: (F, 128) (M の 64 → S の 64、各フレームの係数) → (n, 2) の L/R (±1 の実数)"""
    M = np.concatenate([haar_inv(c) for c in coef[:, :64]])
    S = np.concatenate([haar_inv(c) for c in coef[:, 64:]])
    return np.stack([M + S, M - S], 1) / float(1 << 22)


def infer(net, meta, llr, step, var):
    """窓 WIN フレームを、端 8 フレームを重ねてずらしながら推定。返り値: (E0, net) どちらも (F, 128)"""
    F = len(llr)
    lost = np.zeros(F, bool)
    e0 = np.zeros((F, meta.ncoef))
    est = np.zeros((F, meta.ncoef))
    edge, hop = 8, WIN - 16
    starts = list(range(0, max(F - WIN, 0) + 1, hop))
    if starts[-1] + WIN < F:
        starts.append(F - WIN)
    for t0 in starts:
        sl = slice(t0, t0 + WIN)
        l_, s_, v_, lo_ = (torch.from_numpy(a[sl][None]) for a in (llr.astype(np.float32), step.astype(np.float32), var.astype(np.float32), lost))
        with torch.no_grad():
            x, S, e0n = features(l_, s_, v_, lo_, meta)
            y = (e0n + net(x)) * S
            z = e0n * S
        a = t0 + (0 if t0 == 0 else edge)
        b = t0 + WIN - (0 if t0 + WIN >= F else edge)
        est[a:b] = y[0].numpy()[a - t0:b - t0]
        e0[a:b] = z[0].numpy()[a - t0:b - t0]
    return e0, est


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default=os.path.join(GRACEFM, "eval_audio", "lossless", "Emotional_piano.wav"))
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--esn0", default="8,12,16,24")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "runs", "baseline", "B200.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "demo", "b200"))
    ap.add_argument("--tag", default="piano")
    a = ap.parse_args()
    torch.set_num_threads(2)
    from visqol import VisqolApi
    api = VisqolApi()
    api.create(mode="audio")
    p = PROFILES["B200"]
    lay = Layout(p)
    ck = torch.load(a.ckpt, map_location="cpu")
    net = HaarNetLite(**ck["cfg"]).eval()
    net.load_state_dict(ck["model"])
    x, fs = wavio.read(a.audio)
    x = wavio.resample(x, fs, p.fs)[int(a.start * p.fs):int((a.start + a.seconds) * p.fs)] * 0.5
    xi = wavio.to_pcm24(x)
    n = len(xi) // 64 * 64
    ref = xi[:n].astype(float) / (1 << 23)
    os.makedirs(a.out, exist_ok=True)
    sf.write(os.path.join(a.out, f"{a.tag}_original.wav"), ref, p.fs, subtype="PCM_24")
    nfr = n // 64
    step = steps_of(p, lay, xi[:n], nfr)
    npz_like = {"pos": lay.coef_pos, "coef_of_bit": lay.coef_flat, "group": None}
    # group (成分, 帯域) の番号 (prep_haarnet.dataset_index と同じ並び)
    from prep_haarnet import dataset_index
    _, group = dataset_index(p, lay)
    npz_like["group"] = group
    meta = make_meta(npz_like, p)
    frames = Transmitter(p, 0, 0).run(xi[:n])
    scr = modem.prbs(lay.n_tx)
    n_pre = modem.N_PREAMBLE
    nsym = modem.n_symbols(lay.n_tx, lay.n_strong, p.modulation)
    ref48 = [wavio.resample(ref[:, c], p.fs, 48000) for c in (0, 1)]

    def score(y):
        y = y[:n]
        snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - y) ** 2), 1e-12))
        mos = np.mean([float(api.measure_from_arrays(ref48[c], wavio.resample(y[:, c], p.fs, 48000), sample_rate=48000).moslqo) for c in (0, 1)])
        return snr, mos

    print(f"B200 ステレオ {a.seconds}s。SNR [dB] / ViSQOL (出力: {os.path.abspath(a.out)})\n")
    print(f"{'Es/N0':>6}{'gfm(現行)':>16}{'e0':>16}{'ultra':>16}{'ultra_M':>16}")
    for e in [float(v) for v in a.esn0.split(",")]:
        rng = np.random.default_rng(1)
        rx = [chan.adc(chan.awgn(f / float(1 << modem.SYM_FRAC), e, rng)) for f in frames]
        y_g, _ = receive(p, rx, None, n, 0)
        y_g = wavio.from_pcm24(y_g)
        lags = range(-8, 9)                                                  # 現行受信機は、IIR 切り替え用に出力を数サンプル遅らせている。整数遅れを補償
        best = max(lags, key=lambda L: float(np.sum(ref[16:-16] * np.roll(y_g, L, axis=0)[16:-16])))
        y_g = np.roll(y_g, best, axis=0)
        print(f"  (gfm の出力の遅れ: {-best} サンプル、補償済み)", flush=True)
        llrs, vars_ = [], []
        for sym in rx:
            l, _, v = modem.demodulate(sym[n_pre:n_pre + nsym], lay.n_tx, p.modulation, p.alpha)
            llrs.append(np.where(scr == 1, -l, l)[lay.coef_stream_idx]); vars_.append(v)
        e0, est = infer(net, meta, np.array(llrs), step, np.array(vars_, float))
        est_m = est.copy()
        est_m[:, 64:] = e0[:, 64:]                                          # S は e0 のまま
        outs = {"gfm": y_g, "e0": to_lr(e0, nfr), "ultra": to_lr(est, nfr), "ultra_M": to_lr(est_m, nfr)}
        cells = []
        for k, y in outs.items():
            sf.write(os.path.join(a.out, f"{a.tag}_es{int(e)}_{k}.wav"), np.clip(y[:n], -1, 1), p.fs, subtype="PCM_24")
            s, m = score(y)
            cells.append(f"{s:6.2f} /{m:6.3f}")
        print(f"{e:6.0f}" + "".join(f"{c:>16}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
