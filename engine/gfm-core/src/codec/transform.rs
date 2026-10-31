//! 変換: M/S 分離と Haar 多段分解（送信 `analyze`）、その逆（受信 `synthesize`）。GraceFM `gracefm/codec.py` に相当。浮動小数点。
//! 信号の尺度は満点 = 1.0。M = (L+R)/2、S = (L-R)/2。Haar は直交（1/√2）。帯域係数は `[成分 0=M 1=S][帯域 k][係数 j]`。

use crate::profiles::Profile;

const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// 帯域の並びは [低域近似, 詳細（低→高）]
pub fn haar_forward(x: &[f64], levels: usize) -> Vec<Vec<f64>> {
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

/// L/R → M/S。M = (L+R)/2、S = (L-R)/2
pub fn ms_split(x: &[[f64; 2]]) -> [Vec<f64>; 2] {
    [x.iter().map(|v| (v[0] + v[1]) * 0.5).collect(), x.iter().map(|v| (v[0] - v[1]) * 0.5).collect()]
}

/// M/S → L/R。L = M + S、R = M - S
pub fn ms_to_lr(m: f64, s: f64) -> [f64; 2] {
    [m + s, m - s]
}

/// 送信: PCM ブロック（満点 ±1 の L/R）→ 帯域係数。M/S 分離と Haar 多段分解
pub fn analyze(pcm: &[[f64; 2]], p: &Profile) -> Vec<Vec<Vec<f64>>> {
    ms_split(pcm).iter().map(|x| haar_forward(x, p.levels)).collect()
}

/// 受信: 帯域係数 → PCM ブロック。Haar の逆変換と M/S の逆変換（analyze の逆）
pub fn synthesize(bands: &[Vec<Vec<f64>>]) -> Vec<[f64; 2]> {
    let m = haar_inverse(&bands[0]);
    let s = haar_inverse(&bands[1]);
    (0..m.len()).map(|i| ms_to_lr(m[i], s[i])).collect()
}
