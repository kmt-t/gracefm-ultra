"""欠落区間の前後 (非因果) の文脈から埋める基準手法: AR モデルによる両側補間 (Janssen 法に近い最小二乗)。"""
import numpy as np
from scipy.linalg import solve_toeplitz


def ar_fit(x, order):
    """自己相関法で AR 係数 a (a[0]=1, 予測誤差フィルタ) を求める"""
    x = np.asarray(x, float)
    r = np.array([np.dot(x[:len(x) - k], x[k:]) for k in range(order + 1)])
    r[0] = r[0] * (1 + 1e-9) + 1e-9
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    return np.concatenate([[1.0], a])


def interpolate(x, a, b, order=32, ctx=1024):
    """x: 1次元の信号、[a, b) が欠落。左右 ctx サンプルの文脈から AR を推定して補間した x[a:b] を返す。
    片側しか文脈がなければ、その側だけで外挿に近い形で補間する"""
    n = len(x)
    lo, hi = max(0, a - ctx), min(n, b + ctx)
    left, right = x[lo:a], x[b:hi]
    parts = [p for p in (left, right) if len(p) > order + 1]
    if not parts:
        return np.zeros(b - a)
    r = np.zeros(order + 1)
    for p in parts:
        r += np.array([np.dot(p[:len(p) - k], p[k:]) for k in range(order + 1)])
    r[0] = r[0] * (1 + 1e-9) + 1e-9
    coef = np.concatenate([[1.0], solve_toeplitz(r[:order], -r[1:order + 1])])
    g = b - a
    s0, s1 = max(0, a - order), min(n, b + order)           # 方程式に関わる範囲
    seg = np.array(x[s0:s1], float)
    # 誤差 e_t = Σ_k coef[k] x[t-k]、t は [a, b + order) のうち s1 に収まる範囲 (未知を含むもの)
    ts = np.arange(a, min(b + order, s1))
    k = np.arange(order + 1)
    idx = ts[:, None] - k[None, :] - s0                      # seg の添字
    valid = idx >= 0
    idx_c = np.clip(idx, 0, len(seg) - 1)
    unknown = (idx_c + s0 >= a) & (idx_c + s0 < b) & valid
    C = np.where(valid, coef[None, :], 0.0)
    known = np.where(unknown, 0.0, seg[idx_c]) * C
    e_known = known.sum(1)
    A = np.zeros((len(ts), g))
    rows, cols = np.nonzero(unknown)
    A[rows, idx_c[rows, cols] + s0 - a] = C[rows, cols]
    try:
        u = np.linalg.solve(A.T @ A + 1e-9 * np.eye(g), -A.T @ e_known)
    except np.linalg.LinAlgError:
        return np.zeros(g)
    return u
