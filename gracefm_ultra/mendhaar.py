"""
MendHaarNet: 前段 MendNet (誤り訂正。全フレームの仮数ビットの LLR と刻みを出す) → 後段 HaarStage (その LLR から、音楽的に正しい Haar 係数と、
係数の誤差の怪しさを出す)。前段の定義は mendnet.py、窓は WIN フレーム。
"""
import torch
from torch import nn

from .haarnet import soft_e0
from .mendnet import LLR_SCALE, Q22, STEP_OFFSET, bits_out, build_input_bits

EDGE_FRAMES = 4                    # 窓の両端のフレームは損失から除く
HALF = 64                          # 成分あたりの係数の数


def run_mend(net, per_comp, bit_idx, bit_ok, nbit):
    """前段を両成分に走らせて、後段の入力にする。per_comp[ch]: 成分 ch の窓 (Windows.gather)。bit_idx、bit_ok: mendnet.bit_index の結果 (torch)。
    → (LLR (B,T,nbit) 整数 LLR の尺度、刻み (B,T,128) 満点 2^22 の尺度)"""
    b0 = per_comp[0]
    llr = torch.zeros(b0["coef"].shape[0], b0["coef"].shape[1], nbit, device=b0["coef"].device)
    step = torch.zeros(llr.shape[0], llr.shape[1], 2 * HALF, device=llr.device)
    for ch, b in enumerate(per_comp):
        x, _, _, sf = build_input_bits(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], b["hint_h"], b["hint_q"], b["llr"], b["dstep"])
        l, st = bits_out(net(x), b["llr"], sf, b["dstep"])
        ok = bit_ok[ch]
        llr[:, :, bit_idx[ch][ok]] = (l * LLR_SCALE)[:, :, ok]
        step[:, :, ch * HALF:(ch + 1) * HALF] = Q22 * 2.0 ** (st * STEP_OFFSET - STEP_OFFSET).clamp(-40, 12)
    return llr, step


def features(llr, step, status, meta):
    """後段の入力。llr (B,T,nbit)、step (B,T,ncoef)、status (B,T) 0/1 使える・2 壊れた・3 欠落。
    → (入力 (B,C,T), 係数ごとの尺度 S (B,1,ncoef)、E0/S (B,T,ncoef))。S は窓内の (成分, 帯域) の群ごとの rms"""
    e0, sd = soft_e0(llr, step, meta)
    pw = (e0 ** 2 + step ** 2).sum(1) @ meta.G
    S = ((pw / (llr.shape[1] * meta.count[None, :])).sqrt() + 1e-3) @ meta.G.T
    S = S[:, None, :]
    ls = torch.log2(step.clamp_min(1.0) / S) / 16
    st = torch.stack([status <= 1, status == 2, status == 3], -1).float()
    return torch.cat([llr / 32, e0 / S, sd / S, ls, st], -1).transpose(1, 2), S, e0 / S


class HaarStage(nn.Module):
    """TCN。E0 への補正量 (ncoef) と、(成分, 帯域) の群ごとの誤差の log 分散 (ngroup) を出す。出力層はゼロ初期化 (学習前は E0 そのまま)"""

    def __init__(self, nbit, ncoef, ngroup, ch=128, blocks=8):
        super().__init__()
        dil = [1, 2, 4, 8]
        self.ncoef = ncoef
        self.inp = nn.Conv1d(nbit + 3 * ncoef + 3, ch, 1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(blocks)])
        self.c1 = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=dil[i % 4], padding=dil[i % 4]) for i in range(blocks)])
        self.c2 = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(blocks)])
        self.act = nn.GELU()
        self.out = nn.Conv1d(ch, ncoef + ngroup, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for n, c1, c2 in zip(self.norms, self.c1, self.c2):
            h = h + c2(self.act(c1(n(h))))
        y = self.out(h).transpose(1, 2)
        return y[..., :self.ncoef], y[..., self.ncoef:].clamp(-12, 6)


def estimate(net, llr, step, status, meta):
    """→ (係数 (B,T,ncoef) 満点 2^22 の尺度、S、前段だけの係数 E0、群ごとの log 分散 (B,T,ngroup)（S で割った係数の誤差の分散）)"""
    x, S, e0n = features(llr, step, status, meta)
    corr, logvar = net(x)
    return (e0n + corr) * S, S, e0n * S, logvar


def stage_loss(est, e0, S, logvar, truth, meta, nll_weight=0.05):
    """est、e0、truth: (B,T,ncoef) 満点 2^22 の尺度。成分 (M、S) ごとの相対二乗誤差の平均 (窓の両端 EDGE_FRAMES は除く、下限は観測レベルの −13 dB)
    + 怪しさのガウス NLL (誤差は勾配を流さない)。→ (損失 (B,)、前段だけの相対誤差 (B,)、後段の相対誤差 (B,))"""
    t, w = truth / S, S ** 2
    keep = torch.ones(est.shape[:2], device=est.device)
    keep[:, :EDGE_FRAMES] = 0
    keep[:, -EDGE_FRAMES:] = 0
    k3 = keep[..., None]
    nvalid = keep.sum(1)

    def rel(y):
        out = []
        for sl in (slice(0, HALF), slice(HALF, 2 * HALF)):
            err = (((y / S - t) ** 2) * w * k3)[..., sl].sum((1, 2))
            ref = ((t ** 2) * w * k3)[..., sl].sum((1, 2)) + 0.05 * nvalid * w[..., sl].sum((1, 2)) + 1e-6
            out.append(err / ref)
        return 0.5 * (out[0] + out[1])

    r_est, r_e0 = rel(est), rel(e0)
    e2 = ((est / S - t) ** 2).detach() @ meta.G / meta.count[None, None, :]                      # 群ごとの誤差の分散 (B,T,ngroup)
    nll = (0.5 * (logvar + e2 * torch.exp(-logvar)) * k3).sum((1, 2)) / (nvalid * logvar.shape[2]).clamp_min(1)
    return r_est + nll_weight * nll, r_e0, r_est
