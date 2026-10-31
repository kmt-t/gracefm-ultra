//! 量子化: 帯域ごとのブロック浮動小数点。送信 `quantize_band` / `quantize`（GraceFM `gracefm/codec.py` に相当、浮動小数点）。
//! 受信の逆量子化は軟判定で、decoder.rs の `soft_coeffs`。
//!
//! 1 帯域の係数 c は、共有指数 e（刻み 2^e、満点 = 1.0 に対する値）と、符号つき整数の仮数 m（|m| <= qmax）で c ≈ m·2^e と表す。
//! アッテネータは、区間ごとに 1 bit で刻みを半分にする（仮数の最下位 bit を細かく使う）。指数は、帯域ごとに固定した範囲 [emin, emax] の
//! 添字として送る（ビット列の形式）。

use crate::profiles::{band_level, Profile};

/// 指数の基準（刻みの最小は 2^(1 - FRAC) = 2^-21）
pub const FRAC: i64 = 22;

pub fn qmax_of(b: i64) -> i64 {
    (1 << (b - 1)) - 1
}

/// Y 帯域は未対応なので、負側は -qmax
pub fn qmin_of(b: i64) -> i64 {
    -qmax_of(b)
}

pub fn offset_of(b: i64) -> i64 {
    qmax_of(b)
}

/// 帯域 k・仮数幅 b で固定する指数範囲 [emin, emax]
pub fn exp_range(k: usize, b: i64, p: &Profile) -> (i64, i64) {
    let lev = band_level(k, p.levels) as i64;
    let emax = (lev >> 1) + 3 - b;
    let emin = emax - ((1i64 << p.exp_bits) - 1);
    (emin, emax)
}

/// 1 ブロックぶんの量子化結果。[ch][k]（0 bit の帯域は None）
#[derive(Clone, Debug)]
pub struct BlockData {
    pub mant: Vec<Vec<Option<Vec<i64>>>>,
    pub eidx: Vec<Vec<Option<i64>>>,
    pub attn: Vec<Vec<Option<Vec<i64>>>>,
}

/// peak <= qmax·2^e を満たす最小の整数 e を [emin, emax] に収める
fn shared_exponent(peak: f64, qmax: i64, emin: i64, emax: i64) -> i64 {
    if peak <= 0.0 {
        return emin;
    }
    let q = qmax as f64;
    let mut e = (peak / q).log2().ceil() as i64;
    while q * 2f64.powi(e as i32) < peak {
        e += 1;
    }
    while q * 2f64.powi((e - 1) as i32) >= peak {
        e -= 1;
    }
    e.clamp(emin, emax)
}

/// 帯域 (k, 仮数幅 b) の係数のピーク（絶対値の最大）から、`quantize_band` が選ぶ指数インデックス（クリップ最適化なし）。
/// 副系の復号係数から、主系のヘッダの指数を予測するのに使う
pub fn exponent_for_peak(peak: f64, k: usize, b: i64, p: &Profile) -> i64 {
    let (emin, emax) = exp_range(k, b, p);
    shared_exponent(peak, qmax_of(b), emin.max(1 - FRAC), emax) - emin
}

/// 1 帯域の係数 c → (仮数 m, 指数インデックス, アッテネータ)。アッテネータの区間は profile.attenuator_group で決まる。
/// clip_opt: 指数を 1 つ小さくして（飽和を許して）誤差が減るなら、そちらを採る
pub fn quantize_band(c: &[f64], ch: usize, k: usize, p: &Profile, clip_opt: bool) -> (Vec<i64>, i64, Vec<i64>) {
    let b = p.bits(ch)[k];
    let group = p.attenuator_group(ch, k);
    let (qmax, qmin) = (qmax_of(b), qmin_of(b));
    let (emin, emax) = exp_range(k, b, p);
    let peak = c.iter().fold(0f64, |a, v| a.max(v.abs()));
    let e_floor = emin.max(1 - FRAC);
    let e0 = shared_exponent(peak, qmax, e_floor, emax);
    let gsize = if group > 0 { group } else { c.len() };
    let round = |v: f64, step: f64| -> i64 { ((v / step + 0.5).floor() as i64).clamp(qmin, qmax) };

    let candidate = |e: i64| -> (Vec<i64>, Vec<i64>, f64) {
        let step = 2f64.powi(e as i32);
        let mut m_out = Vec::with_capacity(c.len());
        let mut attn = Vec::new();
        let mut err_total = 0f64;
        for chunk in c.chunks(gsize) {
            let m0: Vec<i64> = chunk.iter().map(|&v| round(v, step)).collect();
            let err0: f64 = chunk.iter().zip(&m0).map(|(&v, &m)| (v - m as f64 * step).powi(2)).sum();
            if group == 0 {
                m_out.extend(m0);
                err_total += err0;
                continue;
            }
            let m1: Vec<i64> = chunk.iter().map(|&v| round(v, step / 2.0)).collect();
            let err1: f64 = chunk.iter().zip(&m1).map(|(&v, &m)| (v - m as f64 * step / 2.0).powi(2)).sum();
            if err1 < err0 {
                attn.push(1);
                m_out.extend(m1);
            } else {
                attn.push(0);
                m_out.extend(m0);
            }
            err_total += err0.min(err1);
        }
        (m_out, attn, err_total)
    };

    let (mut m, mut a, err) = candidate(e0);
    let mut e = e0;
    if clip_opt {
        let e1 = (e0 - 1).max(e_floor);
        let (m1, a1, err1) = candidate(e1);
        if err1 < err {
            e = e1;
            m = m1;
            a = a1;
        }
    }
    (m, e - emin, a)
}

/// analyze の出力 → BlockData（帯域ごとの仮数・指数・アッテネータ）。b = 0 の帯域は None
pub fn quantize(bands: &[Vec<Vec<f64>>], p: &Profile, clip_opt: bool) -> BlockData {
    let nb = p.nbands();
    let mut bd = BlockData { mant: vec![vec![None; nb]; 2], eidx: vec![vec![None; nb]; 2], attn: vec![vec![None; nb]; 2] };
    for (ch, bands_ch) in bands.iter().enumerate() {
        for (k, band) in bands_ch.iter().enumerate().take(nb) {
            if p.bits(ch)[k] == 0 {
                continue;
            }
            let (m, e, a) = quantize_band(band, ch, k, p, clip_opt);
            bd.mant[ch][k] = Some(m);
            bd.eidx[ch][k] = Some(e);
            bd.attn[ch][k] = Some(a);
        }
    }
    bd
}
