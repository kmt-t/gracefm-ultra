"""
MendNet (使えないフレームの補間): 融合後の Haar 係数のフレーム列 (WIN フレーム) と、フレームの状態・誤差分散・ヘッダの信頼度から、
使えない (壊れた・欠落) フレームの係数を非因果に予測する。入力の定義は docs/MENDNET.md。
Haar は直交なので、係数領域の誤差エネルギーは時間領域と同じ。ONNX に出せる演算 (Conv1d, GELU, GroupNorm) だけで組む。
"""
import numpy as np
import torch
from torch import nn

from .plc_ar import interpolate

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
C_IN_HINT = 143                  # v2: 壊れたフレームの手がかり h[64]、q[5] を足す
GAP_SCALE = 8.0
FLOOR = 0.05


class MendNet(nn.Module):
    def __init__(self, c_in=C_IN, ch=192, blocks=12, c_out=64):
        super().__init__()
        dil = [1, 2, 4, 8, 16, 32]
        self.inp = nn.Conv1d(c_in, ch, 1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(blocks)])
        self.c1 = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=dil[i % 6], padding=dil[i % 6]) for i in range(blocks)])
        self.c2 = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(blocks)])
        self.act = nn.GELU()
        self.out = nn.Conv1d(ch, c_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for n, c1, c2 in zip(self.norms, self.c1, self.c2):
            h = h + c2(self.act(c1(n(h))))
        return self.out(h)


def build_input(coef, var, status, gap, comp, hint_h=None, hint_q=None):
    """MendNet の入力 v1 (74 チャンネル)。バッチ B、フレーム T、成分 1 つ。
    coef (B,T,64) 融合後の係数、var (B,T,5) 誤差分散 (情報なしは負)、status (B,T) 0/1 = 使える (1 = 高域欠落)・2 = 壊れた・3 = 欠落、
    gap (B,T) 段 A' の gap、comp (B,) 成分 (M = 0、S = 1)。
    hint_h (B,T,64)、hint_q (B,T,5): 壊れたフレームの手がかり。与えると v2 (143 チャンネル)。
    → (x (B,74 または 143,T), sigma (B,5) 帯域ごとの尺度, usable (B,T) bool)"""
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
    parts = [v, r, s, g[:, :, None], c[:, :, None]]
    if hint_h is not None:
        broken = (status == 2).float()[:, :, None]
        parts += [hint_h * broken, hint_q * broken]
    x = torch.cat(parts, -1)
    return x.transpose(1, 2).contiguous(), sig, usable


Q22 = float(1 << 22)
STEP_OFFSET = 16.0                # 刻みの特徴: (log2(刻み / 2^22) + 16) / 16
LLR_SCALE = 4.0                   # 整数 LLR は自然対数の LLR の 4 倍
STEP_WEIGHT = 0.1
COEF_WEIGHT = 10.0
CORR_MAX = 8.0                    # 受信 LLR があるビットの補正の上限 [nat]。受信で確かなビットを文脈で反転させない
FLOOR_REL = 0.01                  # 係数の相対誤差の分母の下限（窓全体のエネルギーの割合）
FLOOR_ABS = 1e-8                  # 同じく、係数 1 個あたりの絶対の下限（−80 dBFS）。無音の窓で相対誤差が発散しないように


def bit_index(coef_of_bit):
    """仮数ビットの、成分ごとの位置 (npz の coef_of_bit から)。成分 ch の係数は 64 ch .. 64 ch + 63 番。
    → (idx (2, NB) 長い成分に合わせて 0 で詰める, 有効 (2, NB) bool)"""
    cb = np.asarray(coef_of_bit)
    per = [np.nonzero(cb // 64 == ch)[0] for ch in range(2)]
    nb = max(len(v) for v in per)
    idx = np.zeros((2, nb), np.int64)
    ok = np.zeros((2, nb), bool)
    for ch, v in enumerate(per):
        idx[ch, :len(v)] = v
        ok[ch, :len(v)] = True
    return idx, ok


def step_feature(dstep):
    """刻み (…, 64)（0 = 不明）→ (特徴 (…, 64)、既知 (…, 64) float)"""
    known = dstep > 0
    f = (torch.log2(dstep.clamp_min(1.0) / Q22) + STEP_OFFSET) / STEP_OFFSET
    return torch.where(known, f, torch.zeros_like(f)), known.float()


def build_input_bits(coef, var, status, gap, comp, hint_h, hint_q, llr, dstep):
    """MendNet の入力（ビット出力版）。build_input の v2 (143 チャンネル) に、受信 LLR (B,T,NB)（整数。成分のビット順、詰めた分は 0）と、
    復号したヘッダの刻み dstep (B,T,64) を足す。→ (x (B, 143 + NB + 128, T), sig, usable, 刻みの特徴 (B,T,64))"""
    x, sig, usable = build_input(coef, var, status, gap, comp, hint_h, hint_q)
    sf, known = step_feature(dstep)
    x = torch.cat([x, (llr.float() / 32).transpose(1, 2), sf.transpose(1, 2), known.transpose(1, 2)], 1)
    return x.contiguous(), sig, usable, sf


def bits_out(y, llr, sf, dstep):
    """出力 y (B, NB + 64, T) → (自然対数の LLR (B,T,NB)（正 = 0 らしい）、刻みの特徴 (B,T,64))。
    LLR は受信 LLR への補正。受信 LLR があるビットの補正は ±CORR_MAX に収める。刻みは、復号したヘッダがあればそのまま、なければ（dstep = 0）出力"""
    nb = llr.shape[-1]
    y = y.transpose(1, 2)
    corr = torch.where(llr != 0, CORR_MAX * torch.tanh(y[..., :nb] / CORR_MAX), y[..., :nb])
    return llr.float() / LLR_SCALE + corr, torch.where(dstep > 0, sf, y[..., nb:])


def step_of(sf):
    """刻みの特徴 (step_feature の尺度) → 刻み（満点 = 2^22 の尺度）。実際の刻み (満点の 2^-20〜2^-6) より少し広い範囲に収める"""
    return Q22 * 2.0 ** (sf * STEP_OFFSET - STEP_OFFSET).clamp(-22, -4)


def coef_error(llr, step_out, b):
    """出力 (LLR (B,T,NB)（自然対数、正 = 0）、刻みの特徴 (B,T,64)) から作る係数と原音の差 (B,T,64)（満点 = 1.0。量子化の誤差は除く）。
    仮数の期待値と真の仮数の差（Σ (p1 - b) 2^pos）× 推定の刻み + 原音 ×（推定の刻み / 真の刻み − 1）。窓 b は Windows.gather の出力"""
    w = (2.0 ** b["pos"].float())[:, None, :] * b["ok"]
    diff = (torch.sigmoid(-llr) - b["bits"].float()) * w
    e = torch.zeros(diff.shape[0], diff.shape[1], 64, device=diff.device).scatter_add_(2, b["coef_of_bit"][:, None, :].expand_as(diff), diff)
    st = step_of(step_out)
    ratio = torch.where(b["step"] > 0, st / b["step"].clamp_min(1e-9), torch.ones_like(st))
    return e * st / Q22 + b["target"] * (ratio - 1)


def bits_loss(llr_out, step_out, b, usable):
    """b: Windows.gather の出力（bits、ok、step、target、pos、coef_of_bit を使う）。
    損失 = ビットの BCE（有効なビットの平均）+ 刻みの誤差（log2 の Huber）+ log(1 + 係数の相対誤差)（相対誤差 = coef_error の二乗和 /（原音 + 量子化の雑音）の二乗和。推定の刻みを使う）。
    それぞれ、使えるフレームと使えないフレームで別々に平均して足す。
    → (損失 (B,), 使えないフレームの BCE (B,), 使えるフレームの BCE (B,))"""
    okf = b["ok"].float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(-llr_out, b["bits"].float(), reduction="none")      # 正 = 0 なので logit は -LLR
    bce = (bce * okf).sum(2) / okf.sum(2)
    sf_true = step_feature(b["step"])[0]
    st = torch.nn.functional.huber_loss(step_out * STEP_OFFSET, sf_true * STEP_OFFSET, reduction="none").mean(2)
    bad, good = (~usable).float(), usable.float()
    e2, t2 = (coef_error(llr_out, step_out, b) ** 2).sum(2), (b["target"] ** 2).sum(2)                                    # (B,T)
    q2 = ((b["step"] / Q22) ** 2).sum(2) / 12 + FLOOR_ABS * b["step"].shape[-1]                                      # 量子化の雑音 + 絶対の下限 (B,T)

    def mean(v, m):
        return (v * m).sum(1) / m.sum(1).clamp_min(1)

    def rel(m):
        return (e2 * m).sum(1) / (((t2 + q2) * m).sum(1) + FLOOR_REL * t2.sum(1) + 1e-9)

    per = bce + STEP_WEIGHT * st
    coef = torch.log1p(rel(bad)) + torch.log1p(rel(good))                                                              # 小さいときは相対誤差、大外れの窓は対数で抑える
    return mean(per, bad) + mean(per, good) + COEF_WEIGHT * coef, mean(bce, bad), mean(bce, good)


def run_both(net, per_comp, bit_idx, bit_ok, nbit):
    """MendNet を両成分に走らせて、フレームの全ビットに並べる。per_comp[ch]: 成分 ch の窓 (Windows.gather)。bit_idx、bit_ok: bit_index の結果 (torch)。
    → (LLR (B,T,nbit) 整数 LLR の尺度、刻み (B,T,128) 満点 2^22 の尺度)"""
    b0 = per_comp[0]
    llr = torch.zeros(b0["coef"].shape[0], b0["coef"].shape[1], nbit, device=b0["coef"].device)
    step = torch.zeros(llr.shape[0], llr.shape[1], 128, device=llr.device)
    for ch, b in enumerate(per_comp):
        x, _, _, sf = build_input_bits(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], b["hint_h"], b["hint_q"], b["llr"], b["dstep"])
        l, st = bits_out(net(x), b["llr"], sf, b["dstep"])
        ok = bit_ok[ch]
        llr[:, :, bit_idx[ch][ok]] = (l * LLR_SCALE)[:, :, ok]
        step[:, :, ch * 64:(ch + 1) * 64] = step_of(st)
    return llr, step


def compose(y, coef, sig, usable, refine=False):
    """出力 y (B,64,T) を係数に戻す。使えないフレームは y × sigma。使えるフレームは、復号した値 (refine なら、復号した値 + y × sigma)"""
    band = torch.from_numpy(BAND).to(y.device)
    est = y.transpose(1, 2) * sig[:, band][:, None, :]
    return torch.where(usable[:, :, None], coef + est if refine else coef, est)


EDGE = 16
_HINV = {}


def haar_matrix(device):
    """係数 (…,64) → 時間信号 (…,64) の行列 (haar_inv を単位行列に適用したもの)"""
    if device not in _HINV:
        _HINV[device] = torch.from_numpy(haar_inv(np.eye(64))).float().to(device)
    return _HINV[device]


def edge_weight(usable, device):
    """時間領域 (B,T*64) の重み: 使えない区間の、使えるフレームに接する端の EDGE サンプルで 1 から 0 へ下がる"""
    b, t = usable.shape
    ramp = torch.linspace(1, 1 / EDGE, EDGE, device=device)
    pre = torch.zeros(b, t, 64, device=device)              # 直前が使えるフレーム: 先頭側
    post = torch.zeros(b, t, 64, device=device)             # 直後が使えるフレーム: 末尾側
    prev = torch.cat([torch.ones_like(usable[:, :1]), usable[:, :-1]], 1) & ~usable
    nxt = torch.cat([usable[:, 1:], torch.ones_like(usable[:, :1])], 1) & ~usable
    pre[:, :, :EDGE] = prev[:, :, None].float() * ramp
    post[:, :, 64 - EDGE:] = nxt[:, :, None].float() * ramp.flip(0)
    return (pre + post).clamp(max=1).reshape(b, t * 64)


def mend_loss(y, target, sig, usable, coef, edge=1.0, refine=False):
    """target (B,T,64) 原音の係数。使えないフレームの帯域ごとの相対二乗誤差 + 接続部 (使えるフレームとの境界) の時間波形の誤差。
    refine なら、使えるフレームの帯域ごとの相対二乗誤差も足す。→ (損失 (B,), 使えないフレームの帯域ごとの相対誤差 (B,5))"""
    band = torch.from_numpy(BAND).to(y.device)
    sg = sig[:, band][:, None, :]
    t = target / sg
    yn = y.transpose(1, 2)
    est_n = torch.where(usable[:, :, None], coef / sg + yn if refine else coef / sg, yn)

    def relative(mask):
        m3 = mask.float()[:, :, None]
        d, ref = (est_n - t) ** 2 * m3, t ** 2 * m3
        out = []
        for k in range(5):
            m = band == k
            out.append(d[:, :, m].sum((1, 2)) / (ref[:, :, m].sum((1, 2)) + FLOOR * m3.sum((1, 2)) * int(m.sum())).clamp_min(1e-9))
        return torch.stack(out, 1)

    rel = relative(~usable)
    loss = rel.mean(1)
    if refine:
        loss = loss + relative(usable).mean(1)
    e = ((est_n * sg - target) @ haar_matrix(y.device)).reshape(y.shape[0], -1)
    w = edge_weight(usable, y.device)
    power = (sig[:, band] ** 2).mean(1, keepdim=True)                       # 窓の 1 サンプルあたりの電力の目安
    cont = (w * e ** 2).sum(1) / (power[:, 0] * w.sum(1)).clamp_min(1e-12)
    return loss + edge * cont, rel


def unusable_runs(usable):
    """使えないフレームの連続区間 [(a, b), …]"""
    runs, i, n = [], 0, len(usable)
    while i < n:
        if usable[i]:
            i += 1
            continue
        j = i
        while j < n and not usable[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def join_ar(coef, usable, xfade=96):
    """coef (T,64): 使えないフレームを MendNet で埋めた係数。各区間の両端 xfade サンプルを、AR 両側補間 (境界で必ず連続する) へ
    余弦窓で寄せる。区間の中身は MendNet のまま。→ 係数 (T,64)"""
    x = haar_inv(coef).reshape(-1)
    y = x.copy()
    for a, b in unusable_runs(usable):
        a, b = a * BLOCK, b * BLOCK
        ar = interpolate(x, a, b)
        n = min(int(xfade), (b - a) // 2)
        if n <= 0:
            continue
        ramp = 0.5 * (1 + np.cos(np.pi * np.arange(n) / n))
        w = np.zeros(b - a)
        w[:n] = ramp
        w[b - a - n:] = np.maximum(w[b - a - n:], ramp[::-1])
        y[a:b] = w * ar + (1 - w) * x[a:b]
    return haar_fwd(y.reshape(-1, BLOCK))
