#!/usr/bin/env python3
"""
Rust の受信 (engine/gfm-rx: Decoder → MendNet → 判定 → PostNet、ONNX Runtime) が、Python の MendNet と PostNet (PyTorch) と同じ音を出すこと。
同じ seed・シナリオの gfm-tx (シンボル) と gfm-gen --mend (MendNet の入力) は同じ通信路の実現になるので、両方の出力を比べる。
基準: 両者の差が、原音に対して -60 dB 未満 (Rust は逐次処理で、後段の先読みに補間の暫定値を使うぶん、わずかに違う)。

  .venv/bin/python tools/test_rx_onnx.py --mend runs/mendbits_B200_v9.pt --post runs/post_B200.pt
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import rs  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.mendnet import make_meta  # noqa: E402
from gracefm_ultra.postnet import Transform  # noqa: E402
from eval_mendnet import load as load_mend  # noqa: E402
from eval_post import load as load_post, lr, process  # noqa: E402
from gen_post import restore  # noqa: E402
from mend_windows import Windows  # noqa: E402

def ort_lib():
    """.venv の onnxruntime に入っている共有ライブラリ"""
    import glob as _glob
    import onnxruntime
    return sorted(_glob.glob(os.path.join(os.path.dirname(onnxruntime.__file__), "capi", "libonnxruntime.so.*")))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mend", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--models", default=os.path.join(HERE, "..", "engine", "models", "B200"))
    ap.add_argument("--ort", default=os.environ.get("ORT_DYLIB_PATH") or ort_lib())
    a = ap.parse_args()
    p = PROFILES["B200"]
    x = rs.clip(p, seconds=6.0)
    ok = True
    for scen in ((24.0, 0, 0.05, 3.0), (14.0, 1, 0.1, 4.0)):
        with tempfile.TemporaryDirectory() as d:
            raw, npz, sym, out = (os.path.join(d, v) for v in ("x.raw", "m.npz", "a.sym", "o.raw"))
            rs.write_pcm(raw, x)
            s = ",".join(str(float(v)) for v in scen)
            subprocess.run([os.path.join(rs.BIN, "gfm-gen"), "--profile", "B200", "--pcm", raw, "--seed", "1", "--out", npz, "--scen", s], check=True)
            subprocess.run([os.path.join(rs.BIN, "gfm-tx"), "--profile", "B200", "--pcm", raw, "--seed", "1", "--scen", s, "--out", sym], check=True)
            r = subprocess.run([os.path.join(rs.BIN, "gfm-rx"), "--profile", "B200", "--frames", sym, "--out", out, "--models", a.models, "--ort", a.ort],
                               check=True, capture_output=True, text=True)
            y_rs = np.fromfile(out, "<f4").reshape(-1, 2).astype(np.float64)
            z = np.load(npz)
            win = Windows([npz], "cpu")
            meta = make_meta({k: z[k] for k in ("pos", "coef_of_bit")}, p)
            e0, sd = restore(win, 0, load_mend(a.mend, "cpu"), meta)
            T = len(e0) // 4 * 4
            o, g = process(load_post(a.post, "cpu"), Transform(), e0[:T], sd[:T], z["status"][0][:T], "cpu")
        y_py = lr(o["PostNet"])
        n = len(y_py)
        ref = lr(Transform().wave(torch.from_numpy(np.ascontiguousarray(z["truth"][:T].reshape(-1, 2, 64).transpose(1, 0, 2)))).numpy())
        diff = 10 * np.log10(((y_rs[:n] - y_py) ** 2).sum() / (ref ** 2).sum())
        snr = lambda y: 10 * np.log10((ref ** 2).sum() / ((y - ref) ** 2).sum())  # noqa: E731
        print(f"scen {s}: Rust と Python の差 {diff:+.1f} dB (原音比)、SNR Rust {snr(y_rs[:n]):+.2f} / Python {snr(y_py):+.2f} dB、"
              f"判定オン Python {g.mean():.3f}\n  {r.stderr.strip()}")
        ok &= diff < -60
    print("OK" if ok else "NG")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
