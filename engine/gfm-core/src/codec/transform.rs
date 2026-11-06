//! 変換: M/S 分離と多段の帯域分割（送信 `analyze_stream`）、その逆（`synthesize_stream`）。GraceFM `gracefm/codec.py` を浮動小数点にしたもの。
//! 信号の尺度は満点 = 1.0。M = (L+R)/2、S = (L-R)/2。低域の直流の利得は 1 で、直交ではない。帯域係数は `[成分 0=M 1=S][帯域 k][係数 j]`。
//! Haar（S 変換）はブロックの中で完結する。LeGall 5/3 はブロックをまたぎ、フレームの係数は前 15 サンプルと後 16 サンプルにも届く。ストリームの両端は折り返す。

use crate::profiles::{Profile, Transform};

/// 帯域の並びは [低域近似, 詳細（低→高）]。x はストリーム全体（ブロックの整数倍）
pub fn haar_forward(x: &[f64], levels: usize) -> Vec<Vec<f64>> {
    let mut a = x.to_vec();
    let mut details = Vec::new();
    for _ in 0..levels {
        let n = a.len() / 2;
        let d: Vec<f64> = (0..n).map(|i| a[2 * i + 1] - a[2 * i]).collect();
        let s: Vec<f64> = (0..n).map(|i| a[2 * i] + d[i] / 2.0).collect();
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
            x[2 * i] = a[i] - d[i] / 2.0;
            x[2 * i + 1] = x[2 * i] + d[i];
        }
        a = x;
    }
    a
}

/// 帯域の並びは [低域近似, 詳細（低→高）]。x はストリーム全体
pub fn legall_forward(x: &[f64], levels: usize) -> Vec<Vec<f64>> {
    let mut a = x.to_vec();
    let mut details = Vec::new();
    for _ in 0..levels {
        let n = a.len() / 2;
        let d: Vec<f64> = (0..n).map(|i| a[2 * i + 1] - (a[2 * i] + a[2 * (i + 1).min(n - 1)]) / 2.0).collect();
        let s: Vec<f64> = (0..n).map(|i| a[2 * i] + (d[i.saturating_sub(1)] + d[i]) / 4.0).collect();
        details.push(d);
        a = s;
    }
    let mut out = vec![a];
    out.extend(details.into_iter().rev());
    out
}

pub fn legall_inverse(bands: &[Vec<f64>]) -> Vec<f64> {
    let mut a = bands[0].clone();
    for d in &bands[1..] {
        let n = a.len();
        let e: Vec<f64> = (0..n).map(|i| a[i] - (d[i.saturating_sub(1)] + d[i]) / 4.0).collect();
        let mut x = vec![0f64; n * 2];
        for i in 0..n {
            x[2 * i] = e[i];
            x[2 * i + 1] = d[i] + (e[i] + e[(i + 1).min(n - 1)]) / 2.0;
        }
        a = x;
    }
    a
}

fn inverse(bands: &[Vec<f64>], p: &Profile) -> Vec<f64> {
    match p.transform {
        Transform::Haar => haar_inverse(bands),
        Transform::LeGall53 => legall_inverse(bands),
    }
}

/// L/R → M/S。M = (L+R)/2、S = (L-R)/2
pub fn ms_split(x: &[[f64; 2]]) -> [Vec<f64>; 2] {
    [x.iter().map(|v| (v[0] + v[1]) * 0.5).collect(), x.iter().map(|v| (v[0] - v[1]) * 0.5).collect()]
}

/// M/S → L/R。L = M + S、R = M - S
pub fn ms_to_lr(m: f64, s: f64) -> [f64; 2] {
    [m + s, m - s]
}

/// 送信: PCM（満点 ±1 の L/R、ブロックの整数倍）→ フレームごとの帯域係数
pub fn analyze_stream(pcm: &[[f64; 2]], p: &Profile) -> Vec<Vec<Vec<Vec<f64>>>> {
    let forward = match p.transform {
        Transform::Haar => haar_forward,
        Transform::LeGall53 => legall_forward,
    };
    let full = ms_split(pcm).map(|x| forward(&x, p.levels));
    (0..pcm.len() / p.block)
        .map(|f| {
            full.iter().map(|bands| (0..p.nbands()).map(|k| bands[k][f * p.ncoef(k)..(f + 1) * p.ncoef(k)].to_vec()).collect()).collect()
        })
        .collect()
}

/// フレームごとの帯域係数 → PCM（analyze_stream の逆）
pub fn synthesize_stream(frames: &[Vec<Vec<Vec<f64>>>], p: &Profile) -> Vec<[f64; 2]> {
    let join = |ch: usize| -> Vec<f64> {
        let nb = frames[0][ch].len();
        inverse(&(0..nb).map(|k| frames.iter().flat_map(|f| f[ch][k].iter().copied()).collect()).collect::<Vec<Vec<f64>>>(), p)
    };
    let (m, s) = (join(0), join(1));
    (0..m.len()).map(|i| ms_to_lr(m[i], s[i])).collect()
}

/// 係数ごとの、合成の基底のエネルギー（Layout の 1 成分の並び、ブロック長ぶん）。係数の誤差 e が波形に出すエネルギーは e² × これ
pub fn coef_power(p: &Profile) -> Vec<f64> {
    let mut out = Vec::with_capacity(p.block);
    for k in 0..p.nbands() {
        let mut bands: Vec<Vec<f64>> = (0..p.nbands()).map(|i| vec![0.0; 8 * p.ncoef(i)]).collect();
        bands[k][4 * p.ncoef(k)] = 1.0;
        let power: f64 = inverse(&bands, p).iter().map(|v| v * v).sum();
        out.extend(std::iter::repeat_n(power, p.ncoef(k)));
    }
    out
}
