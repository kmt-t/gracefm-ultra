//! 音声符号化: M/S → ブロック内 Haar 多段分解 → 帯域ごとのブロック浮動小数点量子化（GraceFM `gracefm/codec.py` の移植）。

use crate::dsp::*;
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

pub fn dequantize(m: &[i64], eidx: i64, k: usize, b: i64, p: &Profile, ch: usize, attn: &[i64]) -> Vec<i64> {
    let (emin, _) = exp_range(k, b, p);
    let s = eidx + emin + FRAC;
    if attn.is_empty() {
        return m.iter().map(|&v| saturate(shift_signed(v, s))).collect();
    }
    let group = p.attenuator_group(ch, k);
    m.iter().enumerate().map(|(i, &v)| saturate(shift_signed(v, s - attn[i / group]))).collect()
}

/// x: (block) の L/R。符号つき 24 bit 整数
pub fn encode_block(x: &[[i64; 2]], p: &Profile, clip_opt: bool) -> BlockData {
    let ms = [
        x.iter().map(|v| (v[0] + v[1] + 2) >> 2).collect::<Vec<_>>(),
        x.iter().map(|v| (v[0] - v[1] + 2) >> 2).collect::<Vec<_>>(),
    ];
    let nb = p.nbands();
    let mut bd = BlockData {
        mant: vec![vec![None; nb]; 2],
        eidx: vec![vec![None; nb]; 2],
        attn: vec![vec![None; nb]; 2],
    };
    for ch in 0..2 {
        let bands = haar_forward(&ms[ch], p.levels);
        for k in 0..nb {
            let b = p.bits(ch)[k];
            if b == 0 {
                continue;
            }
            let (m, e, a) = quantize(&bands[k], k, b, p, clip_opt, ch);
            bd.mant[ch][k] = Some(m);
            bd.eidx[ch][k] = Some(e);
            bd.attn[ch][k] = Some(a);
        }
    }
    bd
}

/// BlockData → [ch][k] の係数（整数、小数 22 bit）。0 bit の帯域は 0
pub fn decode_coeffs(bd: &BlockData, p: &Profile) -> Vec<Vec<Vec<i64>>> {
    (0..2)
        .map(|ch| {
            (0..p.nbands())
                .map(|k| {
                    let b = p.bits(ch)[k];
                    if b == 0 {
                        vec![0; p.ncoef(k)]
                    } else {
                        dequantize(bd.mant[ch][k].as_ref().unwrap(), bd.eidx[ch][k].unwrap(), k, b, p, ch, bd.attn[ch][k].as_ref().unwrap())
                    }
                })
                .collect()
        })
        .collect()
}
