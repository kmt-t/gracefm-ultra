//! 送信側（量子化・Haar 分解）の整数演算。GraceFM `gracefm/dsp.py` の移植。ビット列の定義に関わるので整数のまま。

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

/// c / 2^s の四捨五入。Python 版（numpy）は s = 0 のとき 1 << -1 が 0 になるので、c をそのまま返す
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
