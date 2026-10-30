"""
HaarNet-lite (試作): M 成分 k=0..2 の仮数bitの LLR と刻みから、係数を非因果に推定する。
生LLRから作る事後平均 E0 に対する補正量を TCN (フレーム方向の畳み込み) で出す。最終層はゼロ初期化なので、学習前は E0 と同じ。
指数・ヘッダは復号済みとする (試作の仮定)。FEC は使わず、保護された上位bitも生LLRのまま入れる。
"""
import numpy as np
import torch
from torch import nn

WIN = 64
NBIT = 140
NCOEF = 16
C_IN = NBIT + 3 * NCOEF + 2


class Meta:
    """bit 属性 (prep_haarnet.py が保存) から、LLR → 係数の事後平均・分散への写像を作る"""

    def __init__(self, pos, coef_of_bit, offsets, qmin, qmax):
        pw = 2.0 ** np.asarray(pos, np.float64)
        w = np.zeros((NBIT, NCOEF))
        w[np.arange(NBIT), coef_of_bit] = pw
        self.W = torch.from_numpy(w).float()
        self.W2 = torch.from_numpy(w ** 2).float()
        self.offset = torch.from_numpy(np.asarray(offsets, np.float64)).float()
        self.qmin = torch.from_numpy(np.asarray(qmin, np.float64)).float()
        self.qmax = torch.from_numpy(np.asarray(qmax, np.float64)).float()


def soft_e0(llr, step, meta):
    """llr: (B, T, 140) 整数LLR (正 = bit 0)、step: (B, T, 16) → 事後平均 E0、標準偏差 (係数の単位)"""
    p1 = torch.sigmoid(-llr / 4.0)
    m = (p1 @ meta.W) - meta.offset
    m = torch.maximum(torch.minimum(m, meta.qmax), meta.qmin)
    v = (p1 * (1 - p1)) @ meta.W2 + 1.0 / 12
    return m * step, v.sqrt() * step


def features(llr, step, var, lost, meta):
    """→ (入力 (B, C_IN, T), 尺度 S (B, 1, 1), E0/S (B, T, 16))。lost のフレームは 0 にしてマスクを足す"""
    keep = (~lost).float()[..., None]
    e0, sd = soft_e0(llr, step, meta)
    e0, sd, step = e0 * keep, sd * keep, step * keep
    band0 = e0[..., :4]
    n = keep.sum((1, 2), keepdim=True).clamp_min(1.0) * 4
    S = ((band0 ** 2).sum((1, 2), keepdim=True) / n).sqrt() + 1e-3
    ls = torch.log2(step.clamp_min(1.0) / S) * keep / 16
    noise = torch.log2(var.clamp_min(1.0))[..., None] / 32 * keep
    x = torch.cat([llr * keep / 32, e0 / S, sd / S, ls, noise, 1 - keep], -1)
    return x.transpose(1, 2), S, e0 / S


class HaarNetLite(nn.Module):
    def __init__(self, ch=128, blocks=8):
        super().__init__()
        dil = [1, 2, 4, 8]
        self.inp = nn.Conv1d(C_IN, ch, 1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(blocks)])
        self.c1 = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=dil[i % 4], padding=dil[i % 4]) for i in range(blocks)])
        self.c2 = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(blocks)])
        self.act = nn.GELU()
        self.out = nn.Conv1d(ch, NCOEF, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for n, c1, c2 in zip(self.norms, self.c1, self.c2):
            h = h + c2(self.act(c1(n(h))))
        return self.out(h).transpose(1, 2)               # (B, T, 16)


def make_meta(npz, profile):
    from gracefm.codec import offset_of, qmax_of, qmin_of
    ncoef = [profile.ncoef(k) for k in range(3)]
    band = np.repeat(np.arange(3), ncoef)
    offs = [offset_of(profile.bits_m[k], profile, "M", k) for k in band]
    qmin = [qmin_of(profile.bits_m[k], profile, "M", k) for k in band]
    qmax = [qmax_of(profile.bits_m[k]) for k in band]
    return Meta(npz["pos"], npz["coef_of_bit"], offs, qmin, qmax)
