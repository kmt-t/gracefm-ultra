//! 変換: M/S 分離と Haar 多段分解。送信 `analyze` は整数（ビット列の定義に関わる。GraceFM `gracefm/codec.py` の移植）、
//! 受信 `synthesize` は浮動小数点（直交な逆 Haar）。係数の尺度は両者で同じ（満点 = 1.0 に対する値）。
//! 帯域係数は `[成分 0=M 1=S][帯域 k][係数 j]`。

use crate::dsp::{mac_sq2_inv, saturate};
use crate::profiles::Profile;

/// 送信（整数）。帯域の並びは [低域近似, 詳細（低→高）]
pub fn haar_forward(x: &[i64], levels: usize) -> Vec<Vec<i64>> {
    let mut a: Vec<i64> = x.iter().map(|&v| saturate(v)).collect();
    let mut details = Vec::new();
    for _ in 0..levels {
        let n = a.len() / 2;
        let mut d = Vec::with_capacity(n);
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            let (e, o) = (a[2 * i], a[2 * i + 1]);
            d.push(mac_sq2_inv(e, o, -1));
            s.push(mac_sq2_inv(e, o, 1));
        }
        details.push(d);
        a = s;
    }
    let mut out = vec![a];
    out.extend(details.into_iter().rev());
    out
}

/// L/R（符号つき 24 bit PCM）→ M/S（小数 22 bit の整数）。M = (L+R)/2、S = (L-R)/2（最後の 2 bit を四捨五入）
pub fn ms_split(x: &[[i64; 2]]) -> [Vec<i64>; 2] {
    [x.iter().map(|v| (v[0] + v[1] + 2) >> 2).collect(), x.iter().map(|v| (v[0] - v[1] + 2) >> 2).collect()]
}

/// 送信: PCM ブロック → 帯域係数（整数、小数 22 bit）
pub fn analyze(pcm: &[[i64; 2]], p: &Profile) -> Vec<Vec<Vec<i64>>> {
    ms_split(pcm).iter().map(|x| haar_forward(x, p.levels)).collect()
}

const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// 浮動小数点の Haar 分解（2 系統の融合で使う）
pub fn haar_forward_f(x: &[f64], levels: usize) -> Vec<Vec<f64>> {
    let mut a = x.to_vec();
    let mut details = Vec::new();
    for _ in 0..levels {
        let n = a.len() / 2;
        let mut d = Vec::with_capacity(n);
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            let (e, o) = (a[2 * i], a[2 * i + 1]);
            d.push((e - o) * SQRT_HALF);
            s.push((e + o) * SQRT_HALF);
        }
        details.push(d);
        a = s;
    }
    let mut out = vec![a];
    out.extend(details.into_iter().rev());
    out
}

pub fn haar_inverse(bands: &[Vec<f64>]) -> Vec<f64> {
    let mut a = bands[0].clone();
    for d in &bands[1..] {
        let mut x = vec![0f64; a.len() * 2];
        for i in 0..a.len() {
            x[2 * i] = (a[i] + d[i]) * SQRT_HALF;
            x[2 * i + 1] = (a[i] - d[i]) * SQRT_HALF;
        }
        a = x;
    }
    a
}

/// 受信: 帯域係数（浮動小数点、満点 ±1）→ L/R（満点 ±1）。L = M + S、R = M - S
pub fn synthesize(bands: &[Vec<Vec<f64>>]) -> Vec<[f64; 2]> {
    let m = haar_inverse(&bands[0]);
    let s = haar_inverse(&bands[1]);
    (0..m.len()).map(|i| [m[i] + s[i], m[i] - s[i]]).collect()
}

/// L/R（満点 ±1）→ M/S。M = (L+R)/2、S = (L-R)/2
pub fn ms_split_f(x: &[[f64; 2]]) -> [Vec<f64>; 2] {
    [x.iter().map(|v| (v[0] + v[1]) * 0.5).collect(), x.iter().map(|v| (v[0] - v[1]) * 0.5).collect()]
}
