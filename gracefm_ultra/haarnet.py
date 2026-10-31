"""
HaarNet-lite (試作): 全成分 (M, S)・全帯域の仮数bitの LLR と刻みから、係数を非因果に推定する。
生LLRから作る事後平均 E0 に対する補正量を TCN (フレーム方向の畳み込み) で出す。最終層はゼロ初期化なので、学習前は E0 と同じ。
指数・ヘッダは復号済みとする (試作の仮定)。FEC は使わず、保護された上位bitも生LLRのまま入れる。
係数の尺度は (成分, 帯域) の群ごとに窓内の rms で正規化する (S は M よりずっと小さいため)。
"""
import numpy as np
import torch
from torch import nn

WIN = 64


class Meta:
    """bit 属性 (prep_haarnet.py が保存) から、LLR → 係数の事後平均・分散への写像を作る"""

    def __init__(self, pos, coef_of_bit, group, offsets, qmin, qmax, group_k=None):
        self.group_k = None if group_k is None else [int(k) for k in group_k]      # 群 (成分, 帯域) ごとの帯域番号 k
        self.coef_of_bit = np.asarray(coef_of_bit, np.int64)
        self.group = np.asarray(group, np.int64)
        self.nbit = len(pos)
        self.ncoef = len(offsets)
        pw = 2.0 ** np.asarray(pos, np.float64)
        w = np.zeros((self.nbit, self.ncoef))
        w[np.arange(self.nbit), coef_of_bit] = pw
        self.W = torch.from_numpy(w).float()
        self.W2 = torch.from_numpy(w ** 2).float()
        self.offset = torch.from_numpy(np.asarray(offsets, np.float64)).float()
        self.qmin = torch.from_numpy(np.asarray(qmin, np.float64)).float()
        self.qmax = torch.from_numpy(np.asarray(qmax, np.float64)).float()
        ng = int(np.max(group)) + 1
        g = np.zeros((self.ncoef, ng))
        g[np.arange(self.ncoef), np.asarray(group)] = 1
        self.G = torch.from_numpy(g).float()                      # (ncoef, 群)
        self.count = self.G.sum(0)                                # 群ごとの係数数

    def to(self, dev):
        """内部のテンソルを dev (cpu / cuda) へ移す"""
        for k in ("W", "W2", "offset", "qmin", "qmax", "G", "count"):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    @property
    def c_in(self):
        return self.nbit + 3 * self.ncoef + 2


def haar_matrix():
    """64 点の直交 Haar 合成行列 H (係数 (.., 64) @ H = サンプル (.., 64))。haar_inv と同じ並び [k=0 (4), k=1 (4), k=2 (8), k=3 (16), k=4 (32)]"""
    from .plcnet import haar_inv
    return np.stack([haar_inv(np.eye(64)[i]) for i in range(64)])


def soft_e0(llr, step, meta):
    """llr: (B, T, nbit) 整数LLR (正 = bit 0)、step: (B, T, ncoef) → 事後平均 E0、標準偏差 (係数の単位)"""
    p1 = torch.sigmoid(-llr / 4.0)
    m = (p1 @ meta.W) - meta.offset
    m = torch.maximum(torch.minimum(m, meta.qmax), meta.qmin)
    v = (p1 * (1 - p1)) @ meta.W2 + 1.0 / 12
    return m * step, v.sqrt() * step


def features(llr, step, var, lost, meta):
    """→ (入力 (B, C_IN, T), 係数ごとの尺度 S (B, 1, ncoef), E0/S (B, T, ncoef))。lost のフレームは 0 にしてマスクを足す"""
    keep = (~lost).float()[..., None]
    e0, sd = soft_e0(llr, step, meta)
    e0, sd, step = e0 * keep, sd * keep, step * keep
    nvalid = keep.sum((1, 2)).clamp_min(1.0)[:, None]                          # (B, 1)
    pw = ((e0 ** 2 + step ** 2).sum(1)) @ meta.G                               # (B, 群) 刻みの二乗を下限として足す
    Sg = (pw / (nvalid * meta.count[None, :])).sqrt() + 1e-3
    S = (Sg @ meta.G.T)[:, None, :]                                            # (B, 1, ncoef)
    ls = torch.log2(step.clamp_min(1.0) / S) * keep / 16
    noise = torch.log2(var.clamp_min(1.0))[..., None] / 32 * keep
    x = torch.cat([llr * keep / 32, e0 / S, sd / S, ls, noise, 1 - keep], -1)
    return x.transpose(1, 2), S, e0 / S


class HaarNetLite(nn.Module):
    def __init__(self, nbit, ncoef, ch=128, blocks=8):
        super().__init__()
        dil = [1, 2, 4, 8]
        self.inp = nn.Conv1d(nbit + 3 * ncoef + 2, ch, 1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(blocks)])
        self.c1 = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=dil[i % 4], padding=dil[i % 4]) for i in range(blocks)])
        self.c2 = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(blocks)])
        self.act = nn.GELU()
        self.out = nn.Conv1d(ch, ncoef, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for n, c1, c2 in zip(self.norms, self.c1, self.c2):
            h = h + c2(self.act(c1(n(h))))
        return self.out(h).transpose(1, 2)               # (B, T, ncoef)


def unet_cfg(meta, widths=None, rounds=1):
    """HaarNetUNet の設定。帯域 k ごとに、その帯域の bit・係数の特徴量の番号 (横入力) と出力する係数の番号を決める。
    段は細かい帯域 (k 大) → 粗い帯域 (k 小) の順。widths は段ごとのチャンネル数 (既定は粗いほど太い)"""
    nb, nc = meta.nbit, meta.ncoef
    ks = sorted(set(meta.group_k), reverse=True)
    gk = np.asarray(meta.group_k)
    cgrp, bgrp = meta.group, meta.group[meta.coef_of_bit]
    in_idx, out_idx = [], []
    for k in ks:
        c = np.where(gk[cgrp] == k)[0]
        b = np.where(gk[bgrp] == k)[0]
        in_idx.append([int(i) for i in np.concatenate([b, nb + c, nb + nc + c, nb + 2 * nc + c, [nb + 3 * nc, nb + 3 * nc + 1]])])
        out_idx.append([int(i) for i in c])
    if widths is None:
        widths = np.linspace(96, 192, len(ks)).round().astype(int).tolist()
    return dict(arch="unet", in_idx=in_idx, out_idx=out_idx, widths=[int(w) for w in widths], nc=nc, nb=nb, rounds=int(rounds))


class _Block(nn.Module):
    """正規化 → 3 点畳み込み (飛び幅 dil) → GELU → 1×1 畳み込み、入力を足す"""

    def __init__(self, ch, dil):
        super().__init__()
        self.n = nn.GroupNorm(1, ch)
        self.c1 = nn.Conv1d(ch, ch, 3, dilation=dil, padding=dil)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.act = nn.GELU()

    def forward(self, h):
        return h + self.c2(self.act(self.c1(self.n(h))))


class HaarNetUNet(nn.Module):
    """帯域方向の U-Net。各帯域の特徴量を、その帯域の段へ横から入れる。
    下り: 細かい帯域 → 粗い帯域へ情報を集める。上り: 粗い帯域の結論を細かい帯域へ戻し、同じ帯域の下りの値をスキップで足す。
    各段は時間方向 (フレーム) の畳み込みで、時間はダウンサンプルしない。最終層は 0 初期化 (学習前は E0 と同じ)"""

    def __init__(self, in_idx, out_idx, widths, nc, nb=0, rounds=1, **_):
        super().__init__()
        L = len(in_idx)
        self.nc, self.nb, self.rounds = nc, nb, rounds
        for j in range(L):
            self.register_buffer(f"i{j}", torch.tensor(in_idx[j], dtype=torch.long))
            self.register_buffer(f"o{j}", torch.tensor(out_idx[j], dtype=torch.long))
        self.side = nn.ModuleList([nn.Conv1d(len(in_idx[j]), widths[j], 1) for j in range(L)])
        self.down = nn.ModuleList([_Block(widths[j], 2 ** (j % 4)) for j in range(L)])
        self.agg = nn.ModuleList([nn.Conv1d(widths[j - 1], widths[j], 1) for j in range(1, L)])        # 細かい → 粗い
        self.back = nn.ModuleList([nn.Conv1d(widths[j + 1], widths[j], 1) for j in range(L - 1)])      # 粗い → 細かい
        self.up = nn.ModuleList([_Block(widths[j], 2 ** ((j + 2) % 4)) for j in range(L)])
        self.out = nn.ModuleList([nn.Conv1d(widths[j], len(out_idx[j]), 1) for j in range(L)])
        for o in self.out:
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)

    def forward(self, x):
        """rounds > 1: 往復を重みを共有して繰り返す。2 回目以降は、入力の E0 の位置に現在の推定 (E0 + 直し分) を戻し、その直し分を足していく。
        下の層の推定が変われば上の層の辻褄が変わり、上の層から下の層の推定がまた変わる、という反復になる"""
        nb, nc = self.nb, self.nc
        corr = self._pass(x)
        for _ in range(self.rounds - 1):
            xr = torch.cat([x[:, :nb], x[:, nb:nb + nc] + corr, x[:, nb + nc:]], 1)
            corr = corr + self._pass(xr)
        return corr.transpose(1, 2)

    def _pass(self, x):
        L = len(self.side)
        d = []
        for j in range(L):
            h = self.side[j](x[:, getattr(self, f"i{j}")])
            if j:
                h = h + self.agg[j - 1](d[-1])
            d.append(self.down[j](h))
        y = [None] * L
        y[L - 1] = self.up[L - 1](d[L - 1])
        for j in range(L - 2, -1, -1):
            y[j] = self.up[j](d[j] + self.back[j](y[j + 1]))
        out = x.new_zeros(x.shape[0], self.nc, x.shape[2])
        for j in range(L):
            out[:, getattr(self, f"o{j}")] = self.out[j](y[j])
        return out


def build_net(cfg):
    """チェックポイントの cfg から網を作る (arch がなければ旧来の HaarNetLite)"""
    if cfg.get("arch") == "unet":
        return HaarNetUNet(**cfg)
    return HaarNetLite(**cfg)


def make_meta(npz, profile):
    """prep_haarnet.py の出力 (npz) とプロファイルから Meta を作る"""
    from gracefm.codec import offset_of, qmax_of, qmin_of
    from gracefm.layout import Layout
    lay = Layout(profile)
    offs, qmin, qmax = [], [], []
    for (ch, k), o in lay.coef_offset.items():
        bits = {"M": profile.bits_m, "S": profile.bits_s}[ch][k]
        n = profile.ncoef(k)
        offs += [offset_of(bits, profile, ch, k)] * n
        qmin += [qmin_of(bits, profile, ch, k)] * n
        qmax += [qmax_of(bits)] * n
    return Meta(npz["pos"], npz["coef_of_bit"], npz["group"], offs, qmin, qmax, group_k=[k for (_, k) in lay.coef_offset])
