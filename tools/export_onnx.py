#!/usr/bin/env python3
"""
受信 (Rust の engine/gfm-rx) が使う ONNX を書き出す。入力の特徴づくりと出力の後処理もグラフに入れて、Rust は観測と状態を並べて渡すだけにする。

- mendnet.onnx: MendNet の窓 (WIN = 100 フレーム、成分 1 つ) のバッチ → 事後平均 e0 と事後標準偏差 sd (満点 1.0)。
  入力: coef (B,100,64) 復号した係数、var (B,100,5) 誤差分散 (情報なしは負)、status (B,100) int64、gap (B,100)、comp (B,) int64 (M = 0、S = 1)、
  hint_h (B,100,64)、hint_q (B,100,5)、llr (B,100,NB) 受信の整数 LLR (成分のビット順、詰めた分は 0)、dstep (B,100,64) 復号したヘッダの刻み (0 = 不明)
- postnet.onnx: PostNet の 1 歩 (逐次処理)。MDCT のフレーム f は波形 [256(f−1), 256(f+1)) にかかる (f = 0 がストリームの先頭)。
  MDCT のフレーム [K, K + F) を処理して、PostNet の出力 [K, K + L) (L = F − 2、先読み 2 フレームのぶん短い) と、音声 [256(K−1), 256(K + L − 1)) を出す。
  入力: e0、sd (2, 4F + 4, 64) M/S の Haar フレーム [4K − 4, 4K + 4F)、on (L,) float (1 = その MDCT フレームを超解像、0 = MendNet のまま)、
  yprev (2, 256, 1) 切り替え後の MDCT のフレーム K − 1、c0〜c9 (2, 256, w_i) 各段の入力の、K より前の w_i フレーム (先頭はすべて 0)。
  出力: audio (2, 256 L) M/S、y (2, 256, L) 切り替え後の MDCT のフレーム [K, K + L)、u0〜u9 各段の入力 (c_i に続けて、この歩のフレーム)。
  次の歩の状態は、確定したフレームの位置 K' で u_i と y から切り出す (`engine/gfm-core/src/post.rs`)。

検証: 同じ入力で PyTorch と ONNX Runtime の出力を比べ、PostNet は逐次処理が全体を一度に通したもの (tools/eval_post.py の process) と一致することも確かめる。

  .venv/bin/python tools/export_onnx.py --mend runs/mendnet_B200.pt --post runs/post_B200.pt --out engine/models/B200
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm.profiles import PROFILES  # noqa: E402
from gracefm_ultra.mendnet import Q22, WIN, bit_index, bits_out, build_input_bits, make_meta, step_of  # noqa: E402
from gracefm_ultra.postnet import A_MAX, LOOKAHEAD, M, Transform, features  # noqa: E402
from eval_mendnet import load as load_mend  # noqa: E402
from eval_post import load as load_post  # noqa: E402

HOP = M // 64                         # MDCT のフレームあたりの Haar フレーム数


class MendGraph(nn.Module):
    def __init__(self, net, meta, idx, ok):
        super().__init__()
        self.net = net
        W = torch.zeros(2, idx.shape[1], 64)
        for ch in range(2):
            W[ch] = meta.W[idx[ch]][:, ch * 64:(ch + 1) * 64] * torch.from_numpy(ok[ch]).float()[:, None]
        self.register_buffer("W", W)
        self.register_buffer("W2", W ** 2)
        for k in ("offset", "qmin", "qmax"):
            self.register_buffer(k, getattr(meta, k).reshape(2, 64))

    def forward(self, coef, var, status, gap, comp, hint_h, hint_q, llr, dstep):
        x, _, _, sf = build_input_bits(coef, var, status, gap, comp, hint_h, hint_q, llr, dstep)
        l, st = bits_out(self.net(x), llr, sf, dstep)
        p1 = torch.sigmoid(-l)
        m = torch.bmm(p1, self.W[comp]) - self.offset[comp][:, None]
        m = torch.maximum(torch.minimum(m, self.qmax[comp][:, None]), self.qmin[comp][:, None])
        v = torch.bmm(p1 * (1 - p1), self.W2[comp]) + 1.0 / 12
        step = step_of(st) / Q22
        return m * step, v.sqrt() * step


def widths(net):
    """各段の入力の、持ち越すフレーム数 (先読みの段は過去 d、因果の段は 2d)"""
    return [d if i < net.lookahead else 2 * d for i, d in enumerate(net.dil)]


class PostStep(nn.Module):
    """PostNet の 1 歩。postnet.PostNet.forward と同じ計算を、段ごとの入力を持ち越して区間ごとに行う"""

    def __init__(self, net):
        super().__init__()
        assert len(net.dil) == 10, "forward の状態の入力は 10 段ぶん"
        self.net, self.tf = net, Transform()

    def forward(self, e0, sd, on, yprev, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
        net, tf = self.net, self.tf
        cache = (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)
        # 分散の下限: features の log10(var + 1e-16) の 1e-16 が、書き出しで落ちる (無音のフレームで -inf になる) ので、先に下限を付ける
        X, V = tf.mdct(tf.wave(e0)), tf.mdct_var(tf.wave_var(sd)).clamp_min(1e-16)      # (2, M, F)
        h = net.inp(features(X, V, torch.arange(2)))
        us = []
        for i, (n, c1, c2) in enumerate(zip(net.norms, net.c1, net.c2)):
            u = torch.cat([cache[i], n(h)], 2)
            us.append(u)
            r = c2(net.act(c1(u)))                                                        # 長さ len(u) − 2d
            h = h[:, :, :r.shape[2]] + r
        y = net.out(h)
        L = y.shape[2]
        Y = 2 * torch.sigmoid(y[:, :M]) * X[:, :, :L] + A_MAX * torch.tanh(y[:, M:]) * V[:, :, :L].sqrt()
        Y = torch.where(on[None, None, :] > 0.5, Y, X[:, :, :L])
        w = tf.imdct(torch.cat([yprev, Y], 2))                                            # [256(K−2), 256(K + L))
        return (w[:, M:w.shape[1] - M], Y, *us)


def stream(step, e0, sd, g, chunk):
    """PostStep を chunk Haar フレームずつ回して (Rust の post.rs と同じ手順。暫定のフレームはない)、全体の音声 (2, T×64) を返す"""
    T = e0.shape[1]

    def seg(x, lo, hi):
        """Haar フレーム [lo, hi)。範囲外は 0"""
        out = torch.zeros(2, hi - lo, 64)
        a, b = max(lo, 0), min(hi, T)
        out[:, a - lo:b - lo] = x[:, a:b]
        return out

    K, yprev = 0, torch.zeros(2, M, 1)
    cache = [torch.zeros(2, 256, w) for w in widths(step.net)]
    out, done = [], 0
    for c in list(range(chunk, T, chunk)) + [T]:
        E = -(-c // HOP) + LOOKAHEAD + 1
        lo, hi = HOP * K - HOP, HOP * E
        gp = np.concatenate([np.ones(HOP, bool), g, np.ones(4 * HOP + 64, bool)])
        L = E - K - LOOKAHEAD
        on = torch.tensor([float(gp[HOP * f:HOP * f + 2 * HOP].all()) for f in range(K, K + L)])
        a, Y, *us = step(seg(e0, lo, hi), seg(sd, lo, hi), on, yprev, *cache)
        s0 = M * (K - 1)
        out.append(a[:, done - s0:64 * c - s0])
        done = 64 * c
        K2 = max(K, c // HOP - LOOKAHEAD)
        cache = [u[:, :, K2 - K:K2 - K + w] for u, w in zip(us, widths(step.net))]
        if K2 > K:
            yprev = Y[:, :, K2 - K - 1:K2 - K]
        K = K2
    return torch.cat(out, 1)


def export(model, args, names, out_names, path, dynamic):
    torch.onnx.export(model, args, path, input_names=names, output_names=out_names, dynamo=True, dynamic_shapes=dynamic, external_data=False)


def check(path, model, args, names):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = s.run(None, {n: a.numpy() for n, a in zip(names, args)})
    with torch.no_grad():
        ref = model(*args)
    ref = ref if isinstance(ref, tuple) else (ref,)
    err = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
    print(f"  {os.path.basename(path)}: ONNX Runtime と PyTorch の最大誤差 {err:.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--mend", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = sorted(glob.glob(os.path.join(a.data, a.profile, "mend_awgn", "val", "*.npz")))[0]
    z = np.load(f)
    idx, ok = bit_index(z["coef_of_bit"])
    meta = make_meta({k: z[k] for k in ("pos", "coef_of_bit")}, PROFILES[a.profile])
    mend = MendGraph(load_mend(a.mend, "cpu"), meta, torch.from_numpy(idx), ok).eval()

    # 実際のデータの窓 2 つ (成分 M と S)
    sc, t0 = 3, 400
    sl = slice(t0, t0 + WIN)
    rows = []
    for ch in range(2):
        c = slice(ch * 64, ch * 64 + 64)
        rows.append([z["dec"][sc, sl, c], z["var"][sc, sl, ch], z["status"][sc, sl].astype(np.int64), z["gap"][sc, sl], np.int64(ch),
                     z["hint_h"][sc, sl, c], z["hint_q"][sc, sl, ch], (z["llr"][sc, sl][:, idx[ch]] * ok[ch]).astype(np.float32),
                     z["dstep"][sc, sl, c]])
    args = tuple(torch.from_numpy(np.stack([rows[0][i], rows[1][i]])) for i in range(9))
    names = ["coef", "var", "status", "gap", "comp", "hint_h", "hint_q", "llr", "dstep"]
    p = os.path.join(a.out, "mendnet.onnx")
    torch.onnx.export(mend, args, p, input_names=names, output_names=["e0", "sd"], dynamic_axes={n: {0: "B"} for n in names + ["e0", "sd"]},
                      opset_version=17, dynamo=False)
    check(p, mend, args, names)

    net = load_post(a.post, "cpu")
    step = PostStep(net).eval()
    ws = widths(net)
    F = 20
    rng = np.random.default_rng(0)
    e0 = torch.from_numpy(rng.normal(0, 0.01, (2, 4 * F + 4, 64)).astype(np.float32))
    args = (e0, torch.full((2, 4 * F + 4, 64), 1e-4), torch.ones(F - LOOKAHEAD), torch.zeros(2, M, 1),
            *[torch.from_numpy(rng.normal(0, 1, (2, 256, w)).astype(np.float32)) for w in ws])
    names = ["e0", "sd", "on", "yprev"] + [f"c{i}" for i in range(len(ws))]
    Fd = torch.export.Dim("F", min=LOOKAHEAD + 2, max=4096)
    dyn = ({1: 4 * Fd + 4}, {1: 4 * Fd + 4}, {0: Fd - LOOKAHEAD}, None, *(None,) * len(ws))
    p = os.path.join(a.out, "postnet.onnx")
    export(step, args, names, ["audio", "y"] + [f"u{i}" for i in range(len(ws))], p, dyn)
    check(p, step, args, names)

    # 逐次処理と、全体を一度に通したものの一致 (実際のデータ、判定の切り替えあり)
    from eval_post import process
    pz = np.load(sorted(glob.glob(os.path.join(a.data, a.profile, "post", "awgn_val", "*.npz")))[0])
    s = int(np.flatnonzero(np.isclose(pz["scen"][:, 0], 22.0))[0])
    T = 3000
    full, g = process(net, Transform(), pz["e0"][s][:T], pz["sd"][s][:T], pz["status"][s][:T], "cpu")
    ms = lambda v: torch.from_numpy(np.ascontiguousarray(v[:T].astype(np.float32).reshape(-1, 2, 64).transpose(1, 0, 2)))  # noqa: E731
    with torch.no_grad():
        y = stream(step, ms(pz["e0"][s]), ms(pz["sd"][s]), g, 50).numpy()
    ref, n = full["PostNet"], (T - 2 * HOP) * 64                                     # 末尾 (後ろの無音の扱いが違う) を除く
    d = 10 * np.log10(((y[:, :n] - ref[:, :n]) ** 2).sum() / (ref[:, :n] ** 2).sum())
    print(f"  逐次処理と一度に通したものの差: {d:+.1f} dB (判定オン {g.mean():.2f})")
    print(f"→ {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
