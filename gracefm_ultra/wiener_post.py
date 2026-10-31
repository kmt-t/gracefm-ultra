"""
Haar 合成のあとの周波数領域ウィナー型ポストフィルタ (実験用、オフライン)。
低域の係数の誤差は、Haar 基底の階段状の形のために広帯域の雑音として漏れる。係数ごとの事後分散 σ_c² が分かっていれば、
その雑音の期待スペクトルは基底の STFT から解析的に求まる:
    E|N(t, f)|² = Σ_c σ_c² |STFT[基底 c をそのブロックの位置に置いたもの](f)|²   (係数の誤差が独立・零平均のとき)
観測の STFT のパワー |Y|² との比から、ウィナー利得 G = ξ / (1 + ξ)、ξ = max(|Y|²/E|N|² - 1, 0) を掛ける。
窓は sqrt-Hann 512 点、ホップ 128 (= 2 ブロック)。ブロック境界に窓が揃うので、基底の表は 8 通りの位置だけ持てばよい。
"""
import numpy as np

from .plcnet import haar_inv

NFFT, HOP, BLOCK = 512, 128, 64
WIN = np.sqrt(np.hanning(NFFT + 1)[:-1])           # 解析窓・合成窓 (sqrt-Hann、75% 重なりで COLA)
_TABLE = []


def basis_table():
    """T[r, c, f] = |rfft(窓 · 基底 c をブロック r (0..7) の位置に置いたもの)|²。形 (8, 64, 257)"""
    if not _TABLE:
        T = np.zeros((NFFT // BLOCK, BLOCK, NFFT // 2 + 1))
        for c in range(BLOCK):
            e = np.zeros(BLOCK)
            e[c] = 1.0
            psi = haar_inv(e)                                       # 係数 c の基底 (64 サンプル)
            for r in range(NFFT // BLOCK):
                seg = np.zeros(NFFT)
                seg[r * BLOCK:(r + 1) * BLOCK] = WIN[r * BLOCK:(r + 1) * BLOCK] * psi
                T[r, c] = np.abs(np.fft.rfft(seg)) ** 2
        _TABLE.append(T)
    return _TABLE[0]


def wiener_postfilter(y, var, floor=0.03, alpha=1.0):
    """y: (n,) 逆 Haar の出力、var: (n/64, 64) 係数の事後分散 (使えない係数は大きな値)。返り値: (処理後の y, 平均利得)"""
    T = basis_table()
    nb = len(y) // BLOCK
    var = np.minimum(np.asarray(var, float)[:nb], 1e30)
    out = np.zeros(len(y) + NFFT)
    norm = np.zeros(len(y) + NFFT)
    gains = []
    for j in range(0, nb - NFFT // BLOCK + 1, HOP // BLOCK):
        s = j * BLOCK
        Y = np.fft.rfft(WIN * y[s:s + NFFT])
        Sn = np.einsum("rc,rcf->f", var[j:j + NFFT // BLOCK], T) * alpha
        xi = np.maximum(np.abs(Y) ** 2 / np.maximum(Sn, 1e-30) - 1.0, 0.0)
        G = np.maximum(xi / (1.0 + xi), floor)
        gains.append(G)
        out[s:s + NFFT] += WIN * np.fft.irfft(Y * G, NFFT)
        norm[s:s + NFFT] += WIN ** 2
    res = y.copy()
    ok = norm[:len(y)] > 1e-6
    res[ok] = out[:len(y)][ok] / norm[:len(y)][ok]
    return res, np.mean(gains, axis=0) if gains else None
