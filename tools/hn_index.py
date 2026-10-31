"""HaarNet の学習データ（Rust の gfm-gen が作る data/<profile>/<sub>/<split>/*.npz）の係数の並びと、gen の coef 配列との対応。"""
import numpy as np


def dataset_index(p, lay):
    """coef_offset 順の係数番号 → 本家 Python の係数配列 (F, 2, block) の平坦な添字"""
    starts = np.concatenate([[0], np.cumsum([p.ncoef(k) for k in range(p.nbands)])])
    idx = np.zeros(lay.n_coef, np.int64)
    group = np.zeros(lay.n_coef, np.int64)
    for gi, ((ch, k), o) in enumerate(lay.coef_offset.items()):
        n = p.ncoef(k)
        ci = 0 if ch == "M" else 1
        idx[o:o + n] = ci * p.block + starts[k] + np.arange(n)
        group[o:o + n] = gi
    return idx, group
