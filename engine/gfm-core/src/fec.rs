//! 誤り検出と系統符号のパリティ（GraceFM `gracefm/fec.py` の送信側の移植）。
//! 畳み込み符号: 拘束長 7、符号化率 1/2、生成多項式 133, 171（8 進）。

pub const K: usize = 7;
const TAPS: [[usize; 5]; 2] = [[0, 2, 3, 5, 6], [0, 1, 2, 3, 6]];

pub const CRC_CORE: (usize, u32) = (12, 0x80F);
pub const CRC_HIGH: (usize, u32) = (10, 0x233);

/// width bit の CRC（MSB から、初期値は全ビット 1）
pub fn crc(bits: &[u8], width: usize, poly: u32) -> Vec<u8> {
    let mask: u32 = (1 << width) - 1;
    let mut reg = mask;
    for &b in bits {
        let top = (reg >> (width - 1)) & 1;
        reg = (reg << 1) & mask;
        if top ^ b as u32 != 0 {
            reg ^= poly;
        }
    }
    (0..width).map(|i| ((reg >> (width - 1 - i)) & 1) as u8).collect()
}

pub fn crc_core(bits: &[u8]) -> Vec<u8> {
    crc(bits, CRC_CORE.0, CRC_CORE.1)
}

pub fn crc_high(bits: &[u8]) -> Vec<u8> {
    crc(bits, CRC_HIGH.0, CRC_HIGH.1)
}

/// 情報ビット x（n）→（パリティ c1（n+6）, 終端の x（6））
pub fn rsc_parity(x: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let n = x.len();
    // c0[t] = x[t] になる入力 u（再帰形）
    let mut u = vec![0u8; n + K - 1];
    for t in 0..n {
        let g = |d: usize| if t >= d { u[t - d] } else { 0 };
        u[t] = x[t] ^ g(2) ^ g(3) ^ g(5) ^ g(6);
    }
    // u[:n] に 6 個の 0 を付けて (133, 171) で符号化する
    let mut c0 = vec![0u8; n + K - 1];
    let mut c1 = vec![0u8; n + K - 1];
    for t in 0..n + K - 1 {
        let at = |d: usize| if t >= d && t - d < n { u[t - d] } else { 0 };
        c0[t] = TAPS[0].iter().fold(0, |a, &d| a ^ at(d));
        c1[t] = TAPS[1].iter().fold(0, |a, &d| a ^ at(d));
    }
    debug_assert_eq!(&c0[..n], x);
    (c1, c0[n..].to_vec())
}
