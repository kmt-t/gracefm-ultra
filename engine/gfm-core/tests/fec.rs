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

#[test]
fn list_viterbi_orders_candidates_and_starts_with_the_best_path() {
    let mut rng = Rng::new(5);
    let x = rng.bits(80);
    let (parity, tail) = rsc_parity(&x);
    // 弱い雑音: 軟値に擾乱を足す
    let noisy = |bits: &[u8], rng: &mut Rng| -> Vec<f64> {
        bits.iter().map(|&b| (if b == 0 { 2.0 } else { -2.0 }) + 3.0 * (rng.uniform() - 0.5)).collect()
    };
    let (xs, ps, ts) = (noisy(&x, &mut rng), noisy(&parity, &mut rng), noisy(&tail, &mut rng));
    let list = rsc_decode_list(&xs, &ps, &ts, 16);
    assert_eq!(list.len(), 16);
    assert_eq!(list[0].0, rsc_decode(&xs, &ps, &ts), "1 番目は通常のビタビ復号と同じ");
    assert!(list.windows(2).all(|w| w[0].1 >= w[1].1), "尤度の高い順");
    let distinct: std::collections::HashSet<_> = list.iter().map(|(b, _)| b.clone()).collect();
    assert_eq!(distinct.len(), list.len(), "候補は互いに異なる");
}

#[test]
fn list_contains_the_truth_when_the_best_path_is_wrong() {
    // 2 か所の情報ビットを強く誤らせると、最尤は誤るが、真の系列は上位の候補に入る
    let mut rng = Rng::new(6);
    let x = rng.bits(60);
    let (parity, tail) = rsc_parity(&x);
    let mut xs = soft(&x, 3.0);
    xs[20] = -xs[20];
    xs[21] = -xs[21];
    let list = rsc_decode_list(&xs, &soft(&parity, 3.0), &soft(&tail, 3.0), 16);
    assert!(list.iter().any(|(b, _)| *b == x), "真の系列が上位 16 候補にない");
}
