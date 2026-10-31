//! FEC（畳み込み符号の系統形のパリティ、ビタビ復号）と CRC。

mod common;
use common::Rng;
use gfm_core::fec::*;

/// 0 → +a、1 → -a の軟値（正 = 0 らしい）
fn soft(bits: &[u8], a: f64) -> Vec<f64> {
    bits.iter().map(|&b| if b == 0 { a } else { -a }).collect()
}

#[test]
fn crc_detects_every_single_bit_error() {
    let mut rng = Rng::new(1);
    let msg = rng.bits(60);
    let good = crc_core(&msg);
    assert_eq!(good.len(), CRC_CORE.0);
    for i in 0..msg.len() {
        let mut m = msg.clone();
        m[i] ^= 1;
        assert_ne!(crc_core(&m), good, "ビット {} の反転を検出できない", i);
        assert_ne!(crc_high(&m), crc_high(&msg));
    }
}

#[test]
fn systematic_parity_round_trip() {
    let mut rng = Rng::new(2);
    for n in [1, 20, 84, 300] {
        let x = rng.bits(n);
        let (parity, tail) = rsc_parity(&x);
        assert_eq!((parity.len(), tail.len()), (n + K - 1, K - 1));
        let got = rsc_decode(&soft(&x, 8.0), &soft(&parity, 8.0), &soft(&tail, 8.0));
        assert_eq!(got, x, "n = {}", n);
    }
}

#[test]
fn viterbi_corrects_scattered_errors() {
    let mut rng = Rng::new(3);
    let x = rng.bits(200);
    let (parity, tail) = rsc_parity(&x);
    let (mut xs, mut ps, ts) = (soft(&x, 8.0), soft(&parity, 8.0), soft(&tail, 8.0));
    // 離れた位置の 8 か所の軟値を、弱い誤りにする
    for i in (10..190).step_by(23) {
        xs[i] = -xs[i].signum() * 1.0;
        ps[i + 3] = -ps[i + 3].signum() * 1.0;
    }
    assert_eq!(rsc_decode(&xs, &ps, &ts), x);
}

#[test]
fn convolutional_code_round_trip() {
    let mut rng = Rng::new(4);
    let x = rng.bits(120);
    let coded = conv_encode(&x);
    assert_eq!(coded.len(), 2 * (x.len() + K - 1));
    let got = viterbi_decode(&soft(&coded, 5.0), x.len());
    assert_eq!(got, x);
}
