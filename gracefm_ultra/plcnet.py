"""
PriorNet (欠落フレームの補間): Haar 係数のフレーム列 (WIN フレーム = 200 ms @32 kHz) と欠落マスクから、
欠落フレームの係数を非因果に予測する。Haar は直交なので、係数領域の誤差エネルギーは時間領域と同じ。
ONNX に出せる演算 (Conv1d, GELU, GroupNorm) だけで組む。入力の形は固定 (1, 65, WIN)。
"""
import numpy as np
import torch
from torch import nn

BLOCK = 64
WIN = 100
COUNTS = [4, 4, 8, 16, 32]
BAND = np.repeat(np.arange(5), COUNTS)


def haar_fwd(x):
    """x: (..., 64) → (..., 64) 係数。並びは gracefm と同じ [k=0 (4), k=1 (4), k=2 (8), k=3 (16), k=4 (32)]"""
    a, det = np.asarray(x, np.float64), []
    for _ in range(4):
        e, o = a[..., 0::2], a[..., 1::2]
        det.append((e - o) / np.sqrt(2))
        a = (e + o) / np.sqrt(2)
    return np.concatenate([a] + det[::-1], -1)


def haar_inv(c):
    c = np.asarray(c, np.float64)
    a, o = c[..., :4], 4
    for n in (4, 8, 16, 32):
        d = c[..., o:o + n]
        o += n
        x = np.empty(a.shape[:-1] + (2 * n,))
        x[..., 0::2] = (a + d) / np.sqrt(2)
        x[..., 1::2] = (a - d) / np.sqrt(2)
        a = x
    return a


class PlcNet(nn.Module):
    def __init__(self, ch=192, blocks=12):
        super().__init__()
        dil = [1, 2, 4, 8, 16, 32]
        self.inp = nn.Conv1d(65, ch, 1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(blocks)])
        self.c1 = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=dil[i % 6], padding=dil[i % 6]) for i in range(blocks)])
        self.c2 = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(blocks)])
        self.act = nn.GELU()
        self.out = nn.Conv1d(ch, 64, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for n, c1, c2 in zip(self.norms, self.c1, self.c2):
            h = h + c2(self.act(c1(n(h))))
        return self.out(h)


def prepare(coef, known):
    """coef: (WIN, 64) 係数、known: (WIN,) bool → (入力 (65, WIN), 帯域ごとの尺度 sigma (5,))。
    尺度 = 既知フレームの帯域ごとの rms。欠落フレームは 0 にして、マスクを 1 チャンネル足す"""
    k = np.asarray(known, bool)
    sig = np.array([np.sqrt(np.mean(coef[k][:, BAND == b] ** 2)) if k.any() else 1.0 for b in range(5)]) + 1e-9
    z = np.where(k[:, None], coef / sig[BAND][None, :], 0.0)
    return np.concatenate([z.T, k[None, :].astype(float)], 0).astype(np.float32), sig


def edge_blend(est, ar, xfade):
    """欠落区間の両端 xfade サンプルで、補間 est を AR 両側補間 ar (境界で必ず連続する) へ余弦窓で寄せる。
    区間の中身は est のまま。境界の不連続 (広帯域の「ざっ」という雑音) を抑える"""
    g = len(est)
    n = min(int(xfade), g // 2)
    if n <= 0:
        return est
    ramp = 0.5 * (1 + np.cos(np.pi * np.arange(n) / n))
    w = np.zeros(g)
    w[:n] = ramp
    w[g - n:] = np.maximum(w[g - n:], ramp[::-1])
    return w * ar + (1 - w) * est


def load_predictor(path, xfade=96):
    """eval_plc.py 用: fn(x, a, b) → x[a:b] の予測 (x は 1 次元の時間信号、a, b は 64 の倍数)。
    xfade: 両端のクロスフェード長 [サンプル] (32 kHz で 96 = 3 ms)。0 なら無し"""
    from .plc_ar import interpolate

    ck = torch.load(path, map_location="cpu")
    net = PlcNet(**ck["cfg"])
    net.load_state_dict(ck["model"])
    net.eval()

    def fn(x, a, b):
        n = len(x) // BLOCK
        fa, fb = a // BLOCK, b // BLOCK
        s = int(np.clip(fa - (WIN - (fb - fa)) // 2, 0, max(n - WIN, 0)))
        coef = haar_fwd(np.asarray(x[s * BLOCK:(s + WIN) * BLOCK], float).reshape(WIN, BLOCK))
        known = np.ones(WIN, bool)
        known[fa - s:fb - s] = False
        inp, sig = prepare(coef, known)
        with torch.no_grad():
            y = net(torch.from_numpy(inp)[None])[0].numpy().T          # (WIN, 64) 正規化済み
        est = haar_inv(y[fa - s:fb - s] * sig[BAND][None, :]).reshape(-1)
        if xfade:
            est = edge_blend(est, interpolate(np.asarray(x, float), a, b), xfade)
        return est

    return fn
