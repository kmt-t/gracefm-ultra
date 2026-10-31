//! 変換: M/S 分離と Haar 多段分解（送信 `analyze`）、その逆（受信 `synthesize`）。GraceFM `gracefm/codec.py` の移植。
//! 帯域係数は `[成分 0=M 1=S][帯域 k][係数 j]`。

use crate::dsp::{mac_sq2_inv, saturate, saturate_bits};
use crate::profiles::Profile;

/// 帯域の並びは [低域近似, 詳細（低→高）]
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

pub fn haar_inverse(bands: &[Vec<i64>]) -> Vec<i64> {
    let mut a = bands[0].clone();
    for d in &bands[1..] {
        let mut x = vec![0i64; a.len() * 2];
        for i in 0..a.len() {
            x[2 * i] = mac_sq2_inv(a[i], d[i], 1);
            x[2 * i + 1] = mac_sq2_inv(a[i], d[i], -1);
        }
        a = x;
    }
    a
}

/// L/R（符号つき 24 bit PCM）→ M/S（小数 22 bit）。M = (L+R)/2、S = (L-R)/2（最後の 2 bit を四捨五入）
pub fn ms_split(x: &[[i64; 2]]) -> [Vec<i64>; 2] {
    [x.iter().map(|v| (v[0] + v[1] + 2) >> 2).collect(), x.iter().map(|v| (v[0] - v[1] + 2) >> 2).collect()]
}

/// M/S（小数 22 bit）→ L/R（符号つき 24 bit PCM）。L = M + S、R = M - S
pub fn ms_to_lr(m: i64, s: i64) -> [i64; 2] {
    [saturate_bits((m + s) << 1, 24), saturate_bits((m - s) << 1, 24)]
}

/// 送信: PCM ブロック → 帯域係数
pub fn analyze(pcm: &[[i64; 2]], p: &Profile) -> Vec<Vec<Vec<i64>>> {
    ms_split(pcm).iter().map(|x| haar_forward(x, p.levels)).collect()
}

/// 受信: 帯域係数 → PCM ブロック
pub fn synthesize(bands: &[Vec<Vec<i64>>]) -> Vec<[i64; 2]> {
    let m = haar_inverse(&bands[0]);
    let s = haar_inverse(&bands[1]);
    (0..m.len()).map(|i| ms_to_lr(m[i], s[i])).collect()
}

// ---------------------------------------------------------------- 浮動小数点（Ultra の本番経路）。係数の尺度は整数版と同じ（±1 = 満点の振幅）

const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

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

pub fn haar_inverse_f(bands: &[Vec<f64>]) -> Vec<f64> {
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

/// 帯域係数（浮動小数点）→ L/R（満点 ±1）。L = M + S、R = M - S
pub fn synthesize_f(bands: &[Vec<Vec<f64>>]) -> Vec<[f64; 2]> {
    let m = haar_inverse_f(&bands[0]);
    let s = haar_inverse_f(&bands[1]);
    (0..m.len()).map(|i| [m[i] + s[i], m[i] - s[i]]).collect()
}

/// L/R（満点 ±1）→ M/S。M = (L+R)/2、S = (L-R)/2
pub fn ms_split_f(x: &[[f64; 2]]) -> [Vec<f64>; 2] {
    [x.iter().map(|v| (v[0] + v[1]) * 0.5).collect(), x.iter().map(|v| (v[0] - v[1]) * 0.5).collect()]
}
