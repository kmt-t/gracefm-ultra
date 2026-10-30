"""評価・学習用の通信路の部品 (../gracefm の channel と組み合わせて使う)。"""
import numpy as np


def loss_mask(nfr, rate, burst, rng):
    """rate: 欠落率。burst: 平均バースト長 [フレーム] (1ならランダム)。Gilbert-Elliott 型"""
    if rate <= 0:
        return np.zeros(nfr, bool)
    p_bg = 1.0 / burst
    p_gb = rate * p_bg / (1 - rate)
    bad, out = False, np.empty(nfr, bool)
    for i in range(nfr):
        out[i] = bad
        bad = (rng.random() >= p_bg) if bad else (rng.random() < p_gb)
    return out
