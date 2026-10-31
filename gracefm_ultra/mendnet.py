"""
MendNet (使えないフレームの補間): 融合後の Haar 係数のフレーム列 (WIN フレーム) と、フレームの状態・誤差分散・ヘッダの信頼度から、
使えない (壊れた・欠落) フレームの係数を非因果に予測する。入力の定義は docs/MENDNET.md。
Haar は直交なので、係数領域の誤差エネルギーは時間領域と同じ。ONNX に出せる演算 (Conv1d, GELU, GroupNorm) だけで組む。
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


C_IN = 74
GAP_SCALE = 8.0
FLOOR = 0.05


class MendNet(nn.Module):
    def __init__(self, c_in=C_IN, ch=192, blocks=12):
        super().__init__()
        dil = [1, 2, 4, 8, 16, 32]
        self.inp = nn.Conv1d(c_in, ch, 1)
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


def build_input(coef, var, status, gap, comp):
    """MendNet の入力 v1 (74 チャンネル)。バッチ B、フレーム T、成分 1 つ。
    coef (B,T,64) 融合後の係数、var (B,T,5) 誤差分散 (情報なしは負)、status (B,T) 0/1 = 使える (1 = 高域欠落)・2 = 壊れた・3 = 欠落、
    gap (B,T) 段 A' の gap、comp (B,) 成分 (M = 0、S = 1)。
    → (x (B,74,T), sigma (B,5) 帯域ごとの尺度, usable (B,T) bool)"""
    band = torch.from_numpy(BAND).to(coef.device)
    usable = status <= 1
    have = (var >= 0) & usable[:, :, None]                                   # (B,T,5) 帯域ごとに値がある
    have64 = have[:, :, band]
    cnt = have64.sum(1).clamp_min(1)                                         # (B,64)
    e = ((coef * have64) ** 2).sum(1) / cnt                                  # 係数ごとの平均電力
    sig = torch.stack([e[:, band == k].mean(1) for k in range(5)], 1).sqrt()
    sig = torch.where(have.any(1), sig, torch.ones_like(sig)).clamp_min(1e-9)   # (B,5)
    v = coef / sig[:, band][:, None, :] * have64
    r = torch.log2(var.clamp_min(1e-30) / sig[:, None, :] ** 2) / 16
    r = torch.where(have, r.clamp(-1, 0.5), torch.full_like(r, 0.5))
    s = torch.stack([usable.float(), (status == 2).float(), (status == 3).float()], -1)
    g = (1 - (gap / GAP_SCALE).clamp(0, 1)) * (status <= 2)
    c = comp.float()[:, None].expand(-1, coef.shape[1])
    x = torch.cat([v, r, s, g[:, :, None], c[:, :, None]], -1)
    return x.transpose(1, 2).contiguous(), sig, usable


def compose(y, coef, sig, usable):
    """出力 y (B,64,T) を係数に戻す。使えるフレームは復号した値、使えないフレームは y × sigma"""
    band = torch.from_numpy(BAND).to(y.device)
    est = y.transpose(1, 2) * sig[:, band][:, None, :]
    return torch.where(usable[:, :, None], coef, est)


def mend_loss(y, target, sig, usable, coef, edge=0.1):
    """target (B,T,64) 原音の係数。使えないフレームの帯域ごとの相対二乗誤差 + 使えないフレームに接する変化量の誤差。
    → (損失 (B,), 帯域ごとの相対誤差 (B,5))"""
    band = torch.from_numpy(BAND).to(y.device)
    bad = (~usable).float()[:, :, None]
    t = target / sig[:, band][:, None, :]
    yn = y.transpose(1, 2)
    d = ((yn - t) ** 2 * bad)
    ref = (t ** 2 * bad)
    rel = []
    for k in range(5):
        m = band == k
        n = bad.sum(1)[:, 0] * int(m.sum())
        rel.append(d[:, :, m].sum((1, 2)) / (ref[:, :, m].sum((1, 2)) + FLOOR * n).clamp_min(1e-9))
    rel = torch.stack(rel, 1)
    v = torch.where(usable[:, :, None], coef / sig[:, band][:, None, :], yn)
    dv, dt = v[:, 1:] - v[:, :-1], t[:, 1:] - t[:, :-1]
    touch = ((~usable[:, 1:]) | (~usable[:, :-1])).float()[:, :, None]
    cont = (((dv - dt) ** 2) * touch).sum((1, 2)) / ((dt ** 2 * touch).sum((1, 2)) + FLOOR * touch.sum((1, 2)) * 64).clamp_min(1e-9)
    return rel.mean(1) + edge * cont, rel
