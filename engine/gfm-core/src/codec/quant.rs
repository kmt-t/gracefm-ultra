//! 量子化: 帯域ごとのブロック浮動小数点。送信 `quantize` は整数（ビット列の定義。GraceFM `gracefm/codec.py` の移植）、受信 `dequantize` は浮動小数点。
//! 受信の `dequantize` は浮動小数点の軟判定: 仮数の各ビットが 1 である確率 p1 から、係数の期待値と誤差分散を求める。

use crate::dsp::*;
use crate::layout::Layout;
use crate::profiles::{band_level, Profile};

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

fn shared_exponent(peak: i64, qmax: i64, emin: i64, emax: i64) -> i64 {
    let ratio = (peak + qmax - 1) / qmax;
    let s = bit_length(ratio.max(1) - 1);
    (s - FRAC).clamp(emin, emax)
}

/// 整数係数 c → (仮数, 指数インデックス, アッテネータ)
pub fn quantize(c: &[i64], k: usize, b: i64, p: &Profile, clip_opt: bool, ch: usize) -> (Vec<i64>, i64, Vec<i64>) {
    let group = p.attenuator_group(ch, k);
    let (qmax, qmin) = (qmax_of(b), qmin_of(b));
    let (emin, emax) = exp_range(k, b, p);
    let peak = c.iter().map(|v| v.abs()).max().unwrap_or(0);
    let e_floor = emin.max(1 - FRAC);
    let e0 = shared_exponent(peak, qmax, e_floor, emax);
    let gsize = if group > 0 { group } else { c.len() };
    let err_shift = (e0 + FRAC - ERR_FRAC_BITS).max(0);

    let candidate = |e: i64| -> (Vec<i64>, Vec<i64>, i64) {
        let s = e + FRAC;
        let mut m_out = Vec::with_capacity(c.len());
        let mut attn = Vec::new();
        let mut err_total = 0i64;
        for chunk in c.chunks(gsize) {
            let m0: Vec<i64> = chunk.iter().map(|&v| round_shift(v, s).clamp(qmin, qmax)).collect();
            let err0: i64 = chunk.iter().zip(&m0).map(|(&v, &m)| sq_err(v - (m << s), err_shift)).sum();
            if group == 0 {
                m_out.extend(m0);
                err_total += err0;
                continue;
            }
            let m1: Vec<i64> = chunk.iter().map(|&v| round_shift(v, s - 1).clamp(qmin, qmax)).collect();
            let err1: i64 = chunk.iter().zip(&m1).map(|(&v, &m)| sq_err(v - (m << (s - 1)), err_shift)).sum();
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

/// 送信: 帯域係数 → BlockData
pub fn quantize_block(bands: &[Vec<Vec<i64>>], p: &Profile, clip_opt: bool) -> BlockData {
    let nb = p.nbands();
    let mut bd = BlockData { mant: vec![vec![None; nb]; 2], eidx: vec![vec![None; nb]; 2], attn: vec![vec![None; nb]; 2] };
    for ch in 0..2 {
        for k in 0..nb {
            let b = p.bits(ch)[k];
            if b == 0 {
                continue;
            }
            let (m, e, a) = quantize(&bands[ch][k], k, b, p, clip_opt, ch);
            bd.mant[ch][k] = Some(m);
            bd.eidx[ch][k] = Some(e);
            bd.attn[ch][k] = Some(a);
        }
    }
    bd
}

/// 学習データ用: 整数の硬判定の逆量子化（小数 22 bit）。送信側の参照で、通信路なしの量子化結果を作る
pub fn reconstruct_hard(bd: &BlockData, p: &Profile) -> Vec<Vec<Vec<i64>>> {
    (0..2)
        .map(|ch| {
            (0..p.nbands())
                .map(|k| {
                    let b = p.bits(ch)[k];
                    if b == 0 {
                        return vec![0; p.ncoef(k)];
                    }
                    let (emin, _) = exp_range(k, b, p);
                    let s = bd.eidx[ch][k].unwrap() + emin + FRAC;
                    let (m, attn, group) = (bd.mant[ch][k].as_ref().unwrap(), bd.attn[ch][k].as_ref().unwrap(), p.attenuator_group(ch, k));
                    m.iter().enumerate().map(|(i, &v)| saturate(shift_signed(v, s - if group > 0 { attn[i / group] } else { 0 }))).collect()
                })
                .collect()
        })
        .collect()
}

/// 誤差分散（線形、満点 1.0 に対する二乗）: [成分][帯域]。情報なしは f64::INFINITY
pub type Rb = [[f64; 5]; 2];

pub struct Estimate {
    pub coef: Vec<Vec<Vec<f64>>>,
    pub rb: [Rb; 2],
}

/// 受信（浮動小数点）: 仮数の各ビットが 1 である確率 p1（0〜1、coef_items の並び）から、係数の期待値と誤差分散を求める。
/// 係数は満点 ±1 の尺度。p1 は厳密な sigmoid でよく、整数版のような Q15 の丸めはない。
pub fn dequantize(bd: &BlockData, lay: &Layout, p1: &[f64], high_ok: bool) -> Estimate {
    let p = &lay.p;
    let nb = p.nbands();
    let mut du = vec![0f64; lay.n_coef];
    let mut var = vec![0f64; lay.n_coef];
    for (j, &cf) in lay.coef_flat.iter().enumerate() {
        let hard = if p1[j] > 0.5 { 1.0 } else { 0.0 };
        let w = (1u64 << lay.coef_pos[j]) as f64;
        du[cf] += (p1[j] - hard) * w;
        var[cf] += p1[j] * (1.0 - p1[j]) * w * w;
    }
    let one = (1u64 << FRAC) as f64;
    let mut coef = vec![vec![Vec::new(); nb]; 2];
    let mut rb = [[[f64::INFINITY; 5]; 2]; 2];
    for ch in 0..2 {
        for k in 0..nb {
            let b = p.bits(ch)[k];
            let n = p.ncoef(k);
            if b == 0 || (k >= p.high_k() && !high_ok) {
                coef[ch][k] = vec![0.0; n];
                continue;
            }
            let o = lay.coef_offset[&(ch, k)];
            let (qmax, qmin) = (qmax_of(b) as f64, qmin_of(b) as f64);
            let mant = bd.mant[ch][k].as_ref().unwrap();
            let attn = bd.attn[ch][k].as_ref().unwrap();
            let group = p.attenuator_group(ch, k);
            let base = bd.eidx[ch][k].unwrap() + exp_range(k, b, p).0 + FRAC;
            let mut c = Vec::with_capacity(n);
            let mut rv = Vec::with_capacity(n);
            for j in 0..n {
                let m = (mant[j] as f64 + du[o + j]).clamp(qmin, qmax);
                let a = if group > 0 { attn[j / group] } else { 0 };
                let step = 2f64.powi((base - a) as i32) / one;
                c.push(m * step);
                rv.push((var[o + j] + 1.0 / 12.0) * step * step);
            }
            let halves = if n >= 2 {
                let h = (n / 2) as f64;
                [rv[..n / 2].iter().sum::<f64>() / h, rv[n / 2..].iter().sum::<f64>() / h]
            } else {
                [rv[0], rv[0]]
            };
            for hh in 0..2 {
                rb[hh][ch][k] = halves[hh];
            }
            coef[ch][k] = c;
        }
    }
    Estimate { coef, rb }
}
