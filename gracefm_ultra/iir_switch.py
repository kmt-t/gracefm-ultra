"""
欠落区間だけ因果の IIR (Butterworth) の出力へ切り替える。IIR は信号全体に常時かけておき (状態は常に最新)、
欠落区間 (前後 約 1 ms のクロスフェード) だけ出力を IIR 側にする。群遅延 (約 2 サンプル) は整数遅れで補償する。
推奨は 4 次、4 kHz (6 素材の掃引で 2〜4 kHz が平坦、5 kHz 以上で劣化)。M 成分、32 kHz。
"""
import numpy as np
from scipy.signal import butter, group_delay, sos2tf, sosfilt

FS = 32000


def iir_switch(y, mask, order=4, fc=4000, xf=33):
    """y: 波形、mask: 欠落区間 (bool)。order 次、fc [Hz] のローパスを、欠落区間だけに切り替える"""
    sos = butter(order, fc / (FS / 2), output="sos")
    b, a = sos2tf(sos)
    _, gd = group_delay((b, a), w=np.array([2 * np.pi * 1000 / FS]))
    D = int(round(float(gd[0])))
    yi = np.roll(sosfilt(sos, y), -D)                                  # 常時計算。群遅延ぶんを補償
    g = np.convolve(mask.astype(float), np.ones(xf) / xf, "same")
    return g * yi + (1 - g) * y
