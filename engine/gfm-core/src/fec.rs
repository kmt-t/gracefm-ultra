//! 誤り検出、系統符号のパリティ（送信）、軟判定ビタビ（受信、浮動小数点）。GraceFM `gracefm/fec.py` の移植。
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

// ---------------------------------------------------------------- 復号（ビタビ）
const NSTATE: usize = 1 << (K - 1);
const G: [u32; 2] = [0o133, 0o171];

/// 情報ビット列を (133, 171) で符号化する（6 ビットの 0 で終端）。符号化ビットの順序は c0[0], c1[0], c0[1], ...
pub fn conv_encode(bits: &[u8]) -> Vec<u8> {
    let n = bits.len() + K - 1;
    let mut out = vec![0u8; 2 * n];
    for t in 0..n {
        for j in 0..2 {
            let mut v = 0u8;
            for &d in &TAPS[j] {
                if t >= d && t - d < bits.len() {
                    v ^= bits[t - d];
                }
            }
            out[2 * t + j] = v;
        }
    }
    out
}

struct Trellis {
    p0: [usize; NSTATE],
    p1: [usize; NSTATE],
    b0: [usize; NSTATE],
    b1: [usize; NSTATE],
    forced: [bool; NSTATE],
}

fn trellis() -> Trellis {
    let out = |s: usize, u: usize, j: usize| -> i64 { (((u << (K - 1)) | s) as u32 & G[j]).count_ones() as i64 & 1 };
    let mut t = Trellis { p0: [0; NSTATE], p1: [0; NSTATE], b0: [0; NSTATE], b1: [0; NSTATE], forced: [false; NSTATE] };
    for ns in 0..NSTATE {
        let u = ns >> (K - 2);
        let p0 = (ns << 1) & (NSTATE - 1);
        let p1 = p0 | 1;
        let idx = |p: usize| -> usize {
            let s0 = 1 - 2 * out(p, u, 0);
            let s1 = 1 - 2 * out(p, u, 1);
            ((1 - s0) + (1 - s1) / 2) as usize
        };
        t.p0[ns] = p0;
        t.p1[ns] = p1;
        t.b0[ns] = idx(p0);
        t.b1[ns] = idx(p1);
        t.forced[ns] = u == 1;
    }
    t
}

/// soft: 長さ 2*(nbits+6) の軟値（正 = 0 らしい、自然対数の LLR）→ 復号ビット（nbits）。経路メトリックは浮動小数点で、正規化だけ行う（飽和なし）
pub fn viterbi_decode(soft: &[f64], nbits: usize) -> Vec<u8> {
    let tr = trellis();
    let l = nbits + K - 1;
    let neg = -1e30f64;
    let mut pm = [neg; NSTATE];
    pm[0] = 0.0;
    let mut choice = vec![[0u8; NSTATE]; l];
    for t in 0..l {
        let (r0, r1) = (soft[2 * t], soft[2 * t + 1]);
        let bm = [r0 + r1, r0 - r1, -r0 + r1, -r0 - r1];
        let mut np = [0f64; NSTATE];
        for ns in 0..NSTATE {
            let m0 = pm[tr.p0[ns]] + bm[tr.b0[ns]];
            let m1 = pm[tr.p1[ns]] + bm[tr.b1[ns]];
            choice[t][ns] = (m1 > m0) as u8;
            np[ns] = m0.max(m1);
        }
        if t >= nbits {
            for ns in 0..NSTATE {
                if tr.forced[ns] {
                    np[ns] = neg;
                }
            }
        }
        let mx = np.iter().cloned().fold(f64::MIN, f64::max);
        for ns in 0..NSTATE {
            pm[ns] = (np[ns] - mx).max(neg);
        }
    }
    let mut s = 0usize;
    let mut bits = vec![0u8; l];
    for t in (0..l).rev() {
        bits[t] = (s >> (K - 2)) as u8;
        s = ((s << 1) & (NSTATE - 1)) | choice[t][s] as usize;
    }
    bits.truncate(nbits);
    bits
}

pub fn rsc_decode(x_soft: &[f64], parity_soft: &[f64], tail_soft: &[f64]) -> Vec<u8> {
    let n = x_soft.len();
    let mut r = vec![0f64; 2 * (n + K - 1)];
    for (i, &v) in x_soft.iter().chain(tail_soft).enumerate() {
        r[2 * i] = v;
    }
    for (i, &v) in parity_soft.iter().enumerate() {
        r[2 * i + 1] = v;
    }
    let u = viterbi_decode(&r, n);
    let c = conv_encode(&u);
    (0..n).map(|i| c[2 * i]).collect()
}
