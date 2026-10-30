"""
float 版の受信器。FEC・CRC の復号は ../gracefm の framing.unpack_recovering をそのまま使い、
その先の仮数の復元を float で行って、係数ごとの事後平均・事後分散を返す。
現行の整数実装 (StreamDecoder.soft_coeffs) と同じ近似 (ビットは独立) だが、事前分布を後から掛けられるように、
丸めずに平均と分散で出す。係数の単位は ../gracefm と同じ (±1 = 2^22) の実数。

LLR は ../gracefm の整数 LLR (小数2 bit、正 = bit 0 寄り) をそのまま受け取る。
"""
import os
import sys

import numpy as np

GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gracefm"))
if GRACEFM not in sys.path:
    sys.path.insert(0, GRACEFM)

from gracefm import framing  # noqa: E402
from gracefm.codec import FRAC, attenuator_expanded, exp_range, qmax_of, qmin_of  # noqa: E402
from gracefm.layout import Layout  # noqa: E402
from gracefm.modem import LLR_FRAC  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402


class FloatRx:
    def __init__(self, profile):
        self.p = PROFILES[profile] if isinstance(profile, str) else profile
        if self.p.has_y:
            raise NotImplementedError("Y方式は未対応")
        self.lay = Layout(self.p)
        self.block = self.p.block

    def _segsum(self, v):
        lay = self.lay
        out = np.zeros(lay.n_coef)
        out[lay._seg_index] = np.add.reduceat(v[lay._seg_order], lay._seg_starts)
        return out

    def frame(self, llr):
        """
        1フレームの LLR (n_tx,) (整数、スクランブル解除済み) → dict
          ok       基本CRCが通ったか (False なら以下は None)
          high_ok  高域CRCが通ったか
          mean     (2, block) 係数の事後平均 [M, S]。使えない係数は 0
          var      (2, block) 事後分散 (係数の単位の二乗)。使えない係数は inf
          p1       仮数bitが1である確率 (coef_items の並び)。上位K bit は確定値 (0/1)
          steps    (2, block) 係数ごとの量子化の刻み (係数の単位)
        """
        lay, p = self.lay, self.p
        llr = np.asarray(llr, np.int64)
        hard = (llr < 0).astype(np.int64)
        bd, hdr = framing.unpack_recovering(lay, llr, hard, None)
        if bd is None:
            return {"ok": False, "high_ok": False, "mean": None, "var": None}
        p1 = 1.0 / (1.0 + np.exp(np.clip(llr[lay.coef_stream_idx] / float(1 << LLR_FRAC), -30, 30)))
        if lay.n_top:
            p1[lay.top_coef_idx] = hdr["top_bits"]           # パリティで訂正した上位K bit は確定値
        hb = (p1 > 0.5).astype(float)
        w = 2.0 ** lay.coef_pos
        du = self._segsum((p1 - hb) * w)
        dv = self._segsum(p1 * (1 - p1) * w * w)
        mean = np.zeros((2, self.block))
        var = np.full((2, self.block), np.inf)
        steps = np.ones((2, self.block))
        bits_of = {"M": p.bits_m, "S": p.bits_s}
        for ci, ch in enumerate(("M", "S")):
            base = 0
            for k in range(p.nbands):
                n = p.ncoef(k)
                b = bits_of[ch][k]
                if b > 0 and (k < p.high_k or hdr["high_ok"]):
                    o = lay.coef_offset[(ch, k)]
                    m = np.clip(bd.mant[ch][k] + du[o:o + n], qmin_of(b, p, ch, k), qmax_of(b))
                    a = attenuator_expanded(bd.attn[ch][k], p.attenuator_group(ch, k), n)
                    s = int(bd.eidx[ch][k]) + exp_range(k, b, p)[0] + FRAC
                    step = 2.0 ** (s - np.asarray(a, float))
                    steps[ci, base:base + n] = step
                    mean[ci, base:base + n] = m * step
                    var[ci, base:base + n] = (dv[o:o + n] + 1.0 / 12) * step ** 2
                base += n
        return {"ok": True, "high_ok": bool(hdr["high_ok"]), "mean": mean, "var": var,
                "p1": p1, "steps": steps}
