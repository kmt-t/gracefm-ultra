//! DSP に依存する整数演算（GraceFM `gracefm/dsp.py` の移植）。numpy の int64 と同じ結果になるようにしてある。

pub const MULT_BITS: u32 = 25;
pub const CONST_BITS: u32 = 18;
pub const MULT_MAX: i64 = (1 << (MULT_BITS - 1)) - 1;
pub const MULT_MIN: i64 = -(1 << (MULT_BITS - 1));
pub const SQ2_SHIFT: u32 = 16;
pub const SQ2_INV: i64 = 46341;
pub const ERR_FRAC_BITS: i64 = 8;

#[inline]
pub fn saturate(v: i64) -> i64 {
    v.clamp(MULT_MIN, MULT_MAX)
}

/// (p + sign*q) / √2。四捨五入して右シフトする
#[inline]
pub fn mac_sq2_inv(p: i64, q: i64, sign: i64) -> i64 {
    let acc = p * SQ2_INV + sign * q * SQ2_INV;
    saturate((acc + (1 << (SQ2_SHIFT - 1))) >> SQ2_SHIFT)
}

/// c / 2^s の四捨五入。numpy 版は s = 0 のとき 1 << -1 が 0 になるので、c をそのまま返す
#[inline]
pub fn round_shift(c: i64, s: i64) -> i64 {
    if s <= 0 {
        c
    } else {
        (c + (1i64 << (s - 1))) >> s
    }
}

/// m * 2^s。s < 0 のときは四捨五入つきの右シフト
#[inline]
pub fn shift_signed(m: i64, s: i64) -> i64 {
    if s >= 0 {
        m << s
    } else {
        round_shift(m, -s)
    }
}

/// n >= 0 のビット長（0 のとき 0）
#[inline]
pub fn bit_length(n: i64) -> i64 {
    if n <= 0 {
        0
    } else {
        64 - (n as u64).leading_zeros() as i64
    }
}

/// 誤差を右シフトして 18 bit 符号つきに切り詰めてから二乗する
#[inline]
pub fn sq_err(err: i64, shift: i64) -> i64 {
    let r = (err >> shift).clamp(-(1 << (CONST_BITS - 1)), (1 << (CONST_BITS - 1)) - 1);
    r * r
}

#[inline]
pub fn saturate_bits(v: i64, bits: u32) -> i64 {
    let lim = (1i64 << (bits - 1)) - 1;
    v.clamp(-lim, lim)
}

// ---------------------------------------------------------------- log と exp（表を使わない）
pub const LN2_Q15: i64 = 22713;
pub const LOG2E_Q8_PER_LLR: i64 = 23637;
pub const ONE: i64 = 1 << 15;

/// 正の整数 x の log2(x) × 2^8
pub fn log2_q8(x: i64) -> i64 {
    let msb = 63 - (x as u64).leading_zeros() as i64;
    let mut m = if msb <= 15 { (x << 15) >> msb } else { x >> (msb - 15) };
    let mut frac = 0i64;
    for _ in 0..8 {
        m = (m * m) >> 15;
        frac <<= 1;
        if m >= 2 * ONE {
            m >>= 1;
            frac |= 1;
        }
    }
    (msb << 8) + frac
}

/// 2^(-d) を Q15 で（d >= 0、Q8）
pub fn pow2_neg_q15(d_q8: i64) -> i64 {
    let t = ((d_q8 & 255) * LN2_Q15) >> 8;
    let mut r = ONE;
    for k in (1..=7).rev() {
        r = ONE - (((t * r) >> 15) / k);
    }
    r >> (d_q8 >> 8).min(31)
}

/// 1 / (1 + 2^d) を Q15 で（d は符号つき、Q8）
pub fn inv1p2pow_q15(d_q8: i64) -> i64 {
    let x = pow2_neg_q15(d_q8.abs().min(16 << 8));
    let w = (x << 15) / (x + ONE);
    if d_q8 >= 0 { w } else { ONE - w }
}

/// 整数 LLR（小数 2 bit、正 = 0 らしい）→ ビットが 1 である確率を Q15 で
pub fn sigmoid_q15(llr: i64) -> i64 {
    inv1p2pow_q15((llr * LOG2E_Q8_PER_LLR + 128) >> 8)
}
