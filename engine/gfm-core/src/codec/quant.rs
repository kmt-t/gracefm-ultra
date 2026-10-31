//! 量子化: 帯域ごとのブロック浮動小数点（送信 `quantize`）と、その逆（受信 `dequantize`）。GraceFM `gracefm/codec.py` の移植。
//! 受信の `dequantize` は軟判定: 仮数の各ビットが 1 である確率 p1（Q15）から、係数の期待値と誤差分散を求める。
//! p1 が 0 か 32768 の確定値なら硬判定の復号と一致する。

use crate::dsp::*;
use crate::layout::Layout;
use crate::profiles::{band_level, Profile};

pub const FRAC: i64 = 22;
/// 誤差分散の「情報なし」（届かない帯域）
pub const INF_LOG: i64 = 1 << 40;
const ONE_TWELFTH_Q15: i64 = 2731;

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

/// 1 帯域の硬判定の逆量子化
pub fn dequantize(m: &[i64], eidx: i64, k: usize, b: i64, p: &Profile, ch: usize, attn: &[i64]) -> Vec<i64> {
    let (emin, _) = exp_range(k, b, p);
    let s = eidx + emin + FRAC;
    if attn.is_empty() {
        return m.iter().map(|&v| saturate(shift_signed(v, s))).collect();
    }
    let group = p.attenuator_group(ch, k);
    m.iter().enumerate().map(|(i, &v)| saturate(shift_signed(v, s - attn[i / group]))).collect()
}

/// 硬判定: BlockData → 帯域係数。0 bit の帯域は 0
pub fn dequantize_hard(bd: &BlockData, p: &Profile) -> Vec<Vec<Vec<i64>>> {
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

/// 誤差分散（対数領域）: [半ブロック][成分][帯域]
pub type Rb = [[i64; 5]; 2];

pub struct Estimate {
    pub coef: Vec<Vec<Vec<i64>>>,
    pub rb: [Rb; 2],
}

/// 受信: 軟判定の逆量子化。hard な仮数 bd に、非保護ビットの事後確率 p1（coef_items の並び、Q15）を反映する。
///   期待値   E[u] = 硬判定の u + Σ (p1 - 硬判定ビット)·2^pos
///   事後分散 Σ p1 (1 - p1)·4^pos
/// 高域の CRC が合わないとき（high_ok = false）は、高域を 0（情報なし）にする。
pub fn dequantize_soft(bd: &BlockData, lay: &Layout, p1: &[i64], high_ok: bool) -> Estimate {
    let p = &lay.p;
    let nb = p.nbands();
    let mut du = vec![0i64; lay.n_coef];
    let mut var = vec![0i64; lay.n_coef];
    for (j, &cf) in lay.coef_flat.iter().enumerate() {
        let hard = (p1[j] > 16384) as i64;
        let w = 1i64 << lay.coef_pos[j];
        du[cf] += (p1[j] - hard * 32768) * w;
        var[cf] += ((p1[j] * (32768 - p1[j])) >> 15) * w * w;
    }
    let mut coef = vec![vec![Vec::new(); nb]; 2];
    let mut rb = [[[INF_LOG; 5]; 2]; 2];
    for ch in 0..2 {
        for k in 0..nb {
            let b = p.bits(ch)[k];
            let n = p.ncoef(k);
            if b == 0 || (k >= p.high_k() && !high_ok) {
                coef[ch][k] = vec![0; n];
                continue;
            }
            let o = lay.coef_offset[&(ch, k)];
            let (qmax, qmin) = (qmax_of(b), qmin_of(b));
            let mant = bd.mant[ch][k].as_ref().unwrap();
            let attn = bd.attn[ch][k].as_ref().unwrap();
            let group = p.attenuator_group(ch, k);
            let base = bd.eidx[ch][k].unwrap() + exp_range(k, b, p).0 + FRAC;
            let mut c = Vec::with_capacity(n);
            let mut rv = Vec::with_capacity(n);
            for j in 0..n {
                let m_q15 = ((mant[j] << 15) + du[o + j]).clamp(qmin << 15, qmax << 15);
                let a = if group > 0 { attn[j / group] } else { 0 };
                c.push(saturate(shift_signed(m_q15, base - a - 15)));
                rv.push((var[o + j] + ONE_TWELFTH_Q15) >> (2 * a));
            }
            let exp_two = 2 * base - 15;
            let halves = if n >= 2 {
                let h = (n / 2) as i64;
                [rv[..n / 2].iter().sum::<i64>() / h, rv[n / 2..].iter().sum::<i64>() / h]
            } else {
                [rv[0], rv[0]]
            };
            for hh in 0..2 {
                rb[hh][ch][k] = log2_q8(halves[hh].max(1)) + (exp_two << 8);
            }
            coef[ch][k] = c;
        }
    }
    Estimate { coef, rb }
}
