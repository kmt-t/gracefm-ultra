"""
帯域分割の合成 (gracefm/codec.py の haar_inverse、legall_inverse を実数で)。kind はプロファイルの transform ("haar" | "legall53")。
係数はフレームごとに 64 個で、帯域 k = 0〜4 の順。直交ではないので、係数の二乗和は波形のエネルギーと一致しない。波形でのエネルギーは、
基底のエネルギー coef_power で重みを付ける。LeGall 5/3 の 1 フレームの係数は、前 15 サンプルと後 16 サンプルにも届く (REACH = 隣の 1 フレーム)。
Haar はブロックの中で完結するが、受信の手順はそろえて同じ REACH で扱う。
"""
import numpy as np

COUNTS = [4, 4, 8, 16, 32]
STRIDES = [16, 16, 8, 4, 2]           # 帯域の係数の間隔 [サンプル]
REACH = 1


def _haar_step(a, d):
    x = np.empty(2 * len(a))
    x[0::2] = a - d / 2
    x[1::2] = x[0::2] + d
    return x


def _legall_step(a, d):
    e = a - (np.concatenate([d[:1], d[:-1]]) + d) / 4
    x = np.empty(2 * len(a))
    x[0::2] = e
    x[1::2] = d + (e + np.concatenate([e[1:], e[-1:]])) / 2
    return x


STEP = {"haar": _haar_step, "legall53": _legall_step}


def inverse(c, kind):
    """係数 (T, 64) → 標本 (T × 64)。LeGall 5/3 のストリームの両端は折り返す"""
    c = np.asarray(c, np.float64)
    bands, o = [], 0
    for n in COUNTS:
        bands.append(c[:, o:o + n].reshape(-1))
        o += n
    a = bands[0]
    for d in bands[1:]:
        a = STEP[kind](a, d)
    return a


def basis(kind):
    """→ (kernels (帯域数, L), before)。kernels[k] は帯域 k の係数の合成の基底で、kernels[k][before] が、その係数の位置の標本"""
    centre, rows = 4 * 64, []
    o = 0
    for n in COUNTS:
        c = np.zeros((8, 64))
        c[4, o] = 1.0
        rows.append(inverse(c, kind))
        o += n
    rows = np.stack(rows)
    on = np.flatnonzero((rows != 0).any(0))
    before = centre - on[0]
    length = max(on[-1] - centre, max(STRIDES)) + before + 1
    return rows[:, centre - before:centre - before + length], int(before)


def coef_power(kind):
    """係数ごとの基底のエネルギー (64,)"""
    return np.repeat((basis(kind)[0] ** 2).sum(1), COUNTS)
