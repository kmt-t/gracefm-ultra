"""
PostNet (MDCT 領域の超解像): 確定した MendNet の係数 (Haar、事後平均 E0) を波形に戻して MDCT にかけ、MendNet の事後標準偏差を
MDCT 領域に移した雑音の地図と合わせて、原音の MDCT 係数を推定する。成分 (M、S) ごとに同じ重みで走らせる。

- MDCT: 窓長 2M = 512、ホップ M = 256 (32 kHz で 8 ms)、サイン窓、直交 (TDAC で完全再構成)。Conv1d / ConvTranspose1d で組む。
- 雑音の地図: Haar の係数の誤差を互いに独立とみなし、分散を線形変換の二乗で移す (時間領域 → MDCT 領域)。
- 網: MDCT フレーム列の TCN (正規化はフレームごと)。先読みは最初の 2 段 (各 1 フレーム) だけで、残りは因果。出力は各ビンのゲイン g と足し込む量 a:
  Y = 2 sigmoid(g) X + A_MAX tanh(a) √(雑音の分散)。足し込みは事後標準偏差の A_MAX 倍まで。初期値は恒等 (Y = X)。
- 対象: 元の音質が良いときだけ。MendNet の事後標準偏差から出す予測の SNR (過去 1 秒の使えるフレームの、係数の二乗和 / 分散の和) が
  GATE_ON 以上でオンになり、GATE_OFF を下回るとオフに戻る (ヒステリシス)。欠落は MendNet が埋めたものを含める。オフのフレームは MendNet の出力のまま (MDCT のフレームごとに切り替え、
  窓の重なりでつながる)。
- 損失: フレーム・帯域 (臨界帯域程度) ごとの相対誤差の対数 (下限 −60 dB)。Haar の量子化の雑音は全帯域に漏れて、弱い高い帯域 (ピアノの倍音など) を
  埋めるので、帯域ごとに原音のエネルギーで正規化して、高い帯域も同じ重みで扱う。加えて複数解像度の対数振幅。
"""
import numpy as np
import torch
from torch import nn

from .haarnet import haar_matrix

M = 256                               # MDCT のビン数 = ホップ
A_MAX = 2.0                           # 足し込む量の上限 (事後標準偏差の倍数)
GATE_WIN = 500                        # 切り替えの判定の平滑化 [Haar フレーム] (過去 1 秒)
GATE_ON = 27.5                        # 予測の SNR がこれ以上になったらオン [dB] (B200 の AWGN で Es/N0 18 = C/N0 68 あたり)
GATE_OFF = 25.5                       # オンのあと、これを下回ったらオフ [dB] (ヒステリシス。閾値の前後での細かい入れ替わりを防ぐ)
FRAME = 64                            # Haar のフレーム長 [サンプル]
X_SCALE = 1e-4                        # 係数の特徴: asinh(X / X_SCALE) / 8
DIL = [1, 1, 2, 4, 8, 1, 2, 4, 8, 16]
LOOKAHEAD = 2                         # 先読みする段の数 (各 1 フレーム)
# 損失の帯域 (MDCT のビンの境目、臨界帯域に近く、2 ビン以上)。帯域ごとの相対誤差なので、弱い高い帯域も低い帯域と同じ重み
BAND_EDGES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 44, 50, 60, 70, 84, 100, 120, 144, 176, 216, M]
BAND_FLOOR = 1e-10                    # 相対誤差の分母の下限 (ビン 1 個あたり、−100 dBFS)
BAND_EPS = 1e-6                       # log(相対誤差 + BAND_EPS) の下限 (−60 dB。コーデックの SNR の上限 50 dB 強より下)
STFT_SIZES = (256, 512, 1024)
STFT_WEIGHT = 0.1
STFT_FLOOR = 1e-5


def gate(e0, sd, status):
    """e0, sd (T, 128)、status (T,) → 超解像をかける Haar フレーム (T,) bool。因果 (過去 GATE_WIN フレーム)。
    GATE_ON 以上でオン、オンの間は GATE_OFF を下回るまでオン。先頭の GATE_WIN フレームは過去が揃わないのでオフ"""
    u = (status <= 1).astype(np.float64)
    sig = np.cumsum((e0.astype(np.float64) ** 2).sum(1) * u)
    var = np.cumsum((sd.astype(np.float64) ** 2).sum(1) * u)
    lag = lambda c: np.concatenate([np.zeros(GATE_WIN), c[:-GATE_WIN]])[:len(c)]  # noqa: E731
    s, v = sig - lag(sig), var - lag(var)
    snr = 10 * np.log10(np.maximum(s, 1e-30) / np.maximum(v, 1e-30))
    out, on = np.zeros(len(s), bool), False
    for t in range(GATE_WIN, len(s)):
        on = s[t] > 0 and snr[t] >= (GATE_OFF if on else GATE_ON)
        out[t] = on
    return out


def mdct_gate(g, F, pad):
    """Haar フレームの判定 g (T,) → MDCT のフレームの判定 (F,)。窓がかかる Haar フレームが全部オンのとき。
    pad: 波形の先頭に足した Haar フレーム数 (MDCT のフレーム f は、足した後の Haar フレーム [4f, 4f + 8) にかかる)"""
    k = M // FRAME
    gp = np.concatenate([np.ones(pad, bool), g, np.ones(pad, bool)])
    return np.array([gp[k * f:k * f + 2 * k].all() for f in range(F)])


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
        return 2 * torch.sigmoid(y[:, :M]) * X + A_MAX * torch.tanh(y[:, M:]) * var.sqrt()


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
    """フレーム・帯域ごとの log(誤差 / (原音 + 下限) + BAND_EPS) の平均 (N,)。Y, T (N, M, F)。
    相対誤差が小さくても 1 dB ごとに同じだけ下がるので、きれいな入力の精度も詰める"""
    out = 0
    for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        e = ((Y[:, a:b] - T[:, a:b]) ** 2).sum(1)
        t = (T[:, a:b] ** 2).sum(1) + BAND_FLOOR * (b - a)
        out = out + torch.log(e / t + BAND_EPS).mean(1)
    return out / (len(BAND_EDGES) - 1)


def post_loss(tf, Y, T, skip):
    """Y, T (N, M, F) 推定と原音の MDCT。先頭 skip フレーム (因果の段の文脈が足りない) は除く → 損失 (N,)"""
    Y, T = Y[:, :, skip:], T[:, :, skip:]
    y, t = tf.imdct(Y)[:, M:-M], tf.imdct(T)[:, M:-M]
    return band_loss(Y, T) + STFT_WEIGHT * stft_loss(y, t)
