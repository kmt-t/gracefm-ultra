"""
PostNet (MDCT 領域の超解像): 確定した MendNet の係数 (Haar、事後平均 E0) を波形に戻して MDCT にかけ、MendNet の事後標準偏差を
MDCT 領域に移した雑音の地図と合わせて、原音の MDCT 係数を推定する。成分 (M、S) ごとに同じ重みで走らせる。

- MDCT: 窓長 2M = 512、ホップ M = 256 (32 kHz で 8 ms)、サイン窓、直交 (TDAC で完全再構成)。Conv1d / ConvTranspose1d で組む。
- 雑音の地図: Haar の係数の誤差を互いに独立とみなし、分散を線形変換の二乗で移す (時間領域 → MDCT 領域)。
- 網: MDCT フレーム列の TCN (正規化はフレームごと)。先読みは最初の 2 段 (各 1 フレーム) だけで、残りは因果。出力は各ビンのゲイン g と足し込む量 a:
  Y = 2 sigmoid(g) X + a √(雑音の分散)。初期値は恒等 (Y = X)。
- 損失: フレーム・帯域 (臨界帯域程度) ごとの相対誤差の対数。Haar の量子化の雑音は全帯域に漏れて、弱い高い帯域 (ピアノの倍音など) を
  埋めるので、帯域ごとに原音のエネルギーで正規化して、高い帯域も同じ重みで扱う。加えて複数解像度の対数振幅。
"""
import numpy as np
import torch
from torch import nn

from .haarnet import haar_matrix

M = 256                               # MDCT のビン数 = ホップ
FRAME = 64                            # Haar のフレーム長 [サンプル]
X_SCALE = 1e-4                        # 係数の特徴: asinh(X / X_SCALE) / 8
DIL = [1, 1, 2, 4, 8, 1, 2, 4, 8, 16]
LOOKAHEAD = 2                         # 先読みする段の数 (各 1 フレーム)
# 損失の帯域 (MDCT のビンの境目、臨界帯域に近く、2 ビン以上)。帯域ごとの相対誤差なので、弱い高い帯域も低い帯域と同じ重み
BAND_EDGES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 44, 50, 60, 70, 84, 100, 120, 144, 176, 216, M]
BAND_FLOOR = 1e-10                    # 相対誤差の分母の下限 (ビン 1 個あたり、−100 dBFS)
STFT_SIZES = (256, 512, 1024)
STFT_WEIGHT = 0.1
STFT_FLOOR = 1e-5


def mdct_basis():
    """(M, 1, 2M) の MDCT 基底 (サイン窓込み、直交)"""
    n, k = np.arange(2 * M), np.arange(M)
    w = np.sin(np.pi * (n + 0.5) / (2 * M))
    b = np.sqrt(2.0 / M) * w[None, :] * np.cos(np.pi / M * (n[None, :] + 0.5 + M / 2) * (k[:, None] + 0.5))
    return torch.from_numpy(b[:, None, :]).float()


class Transform(nn.Module):
    """Haar の係数 ↔ 波形 ↔ MDCT。分散も移す"""

    def __init__(self):
        super().__init__()
        h = torch.from_numpy(haar_matrix()).float()
        self.register_buffer("H", h)
        self.register_buffer("H2", h ** 2)
        b = mdct_basis()
        self.register_buffer("B", b)
        self.register_buffer("B2", b ** 2)

    def wave(self, coef):
        """(N, T, 64) → (N, T × 64)"""
        return (coef @ self.H).reshape(coef.shape[0], -1)

    def wave_var(self, sd):
        """事後標準偏差 (N, T, 64) → 時間領域の分散 (N, T × 64)"""
        return ((sd ** 2) @ self.H2).reshape(sd.shape[0], -1)

    def mdct(self, x):
        """(N, L) → (N, M, L / M − 1)"""
        return nn.functional.conv1d(x[:, None, :], self.B, stride=M)

    def mdct_var(self, v):
        return nn.functional.conv1d(v[:, None, :], self.B2, stride=M)

    def imdct(self, X):
        """(N, M, F) → (N, (F + 1) M)。両端 M サンプルは重ならないので使わない"""
        return nn.functional.conv_transpose1d(X, self.B, stride=M)[:, 0]


class FrameNorm(nn.Module):
    """フレームごとに、チャンネル方向で正規化する (時間方向の統計を使わないので、因果で、系列の長さによらない)"""

    def __init__(self, ch):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, ch, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = ((x - m) ** 2).mean(1, keepdim=True)
        return (x - m) / torch.sqrt(v + 1e-5) * self.g + self.b


def features(X, var, comp):
    """→ (N, 2M + 1, F)"""
    fx = torch.asinh(X / X_SCALE) / 8
    fv = torch.log10(var + 1e-16) / 8 + 1
    c = comp.float()[:, None, None].expand(-1, 1, X.shape[2])
    return torch.cat([fx, fv, c], 1)


class PostNet(nn.Module):
    def __init__(self, ch=256, dil=DIL, lookahead=LOOKAHEAD):
        super().__init__()
        self.dil, self.lookahead = list(dil), lookahead
        self.inp = nn.Conv1d(2 * M + 1, ch, 1)
        self.norms = nn.ModuleList([FrameNorm(ch) for _ in dil])
        self.c1 = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=d) for d in dil])
        self.c2 = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in dil])
        self.act = nn.GELU()
        self.out = nn.Conv1d(ch, 2 * M, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, X, var, comp):
        """X, var (N, M, F) → Y (N, M, F)"""
        h = self.inp(features(X, var, comp))
        for i, (n, c1, c2) in enumerate(zip(self.norms, self.c1, self.c2)):
            d = self.dil[i]
            pad = (d, d) if i < self.lookahead else (2 * d, 0)
            h = h + c2(self.act(c1(nn.functional.pad(n(h), pad))))
        y = self.out(h)
        return 2 * torch.sigmoid(y[:, :M]) * X + y[:, M:] * var.sqrt()


def receptive_past():
    """因果の段が見る過去のフレーム数"""
    return sum(2 * d for d in DIL[LOOKAHEAD:]) + LOOKAHEAD


def stft_loss(y, t):
    """複数の解像度の対数振幅の L1 (N,) 。y, t (N, L)"""
    out = 0
    for n in STFT_SIZES:
        win = torch.hann_window(n, device=y.device)
        sy = torch.stft(y, n, n // 4, window=win, return_complex=True).abs()
        st = torch.stft(t, n, n // 4, window=win, return_complex=True).abs()
        out = out + (torch.log(sy + STFT_FLOOR) - torch.log(st + STFT_FLOOR)).abs().mean((1, 2))
    return out / len(STFT_SIZES)


def band_loss(Y, T):
    """フレーム・帯域ごとの log(1 + 誤差 / (原音 + 下限)) の平均 (N,)。Y, T (N, M, F)"""
    out = 0
    for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        e = ((Y[:, a:b] - T[:, a:b]) ** 2).sum(1)
        t = (T[:, a:b] ** 2).sum(1) + BAND_FLOOR * (b - a)
        out = out + torch.log1p(e / t).mean(1)
    return out / (len(BAND_EDGES) - 1)


def post_loss(tf, Y, T, skip):
    """Y, T (N, M, F) 推定と原音の MDCT。先頭 skip フレーム (因果の段の文脈が足りない) は除く → 損失 (N,)"""
    Y, T = Y[:, :, skip:], T[:, :, skip:]
    y, t = tf.imdct(Y)[:, M:-M], tf.imdct(T)[:, M:-M]
    return band_loss(Y, T) + STFT_WEIGHT * stft_loss(y, t)
