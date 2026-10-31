//! 変調と復調（QPSK と階層 16QAM）。

mod common;
use common::Rng;
use gfm_core::modem::*;

fn modems() -> Vec<(&'static str, Modem)> {
    vec![("qpsk", Modem::new(false, 1)), ("qam16h", Modem::new(true, 5))]
}

#[test]
fn noise_free_demodulation_recovers_the_bits() {
    let mut rng = Rng::new(1);
    for (name, m) in modems() {
        let nsym = 300;
        let bits = rng.bits(nsym * m.bits_per_symbol());
        let sym = m.modulate(&bits, nsym);
        assert_eq!(sym.len(), N_PREAMBLE + nsym, "{}", name);
        let (llr, hard, var) = m.demodulate(&sym[N_PREAMBLE..], bits.len());
        assert_eq!(hard, bits, "{}: 硬判定", name);
        assert!(llr.iter().zip(&bits).all(|(&l, &b)| (l > 0.0) == (b == 0)), "{}: LLR の符号", name);
        assert!(var <= 1e-11, "{}: 雑音なしの分散 {}", name, var);
    }
}

#[test]
fn symbols_have_unit_average_energy() {
    let mut rng = Rng::new(2);
    for (name, m) in modems() {
        let nsym = 20000;
        let bits = rng.bits(nsym * m.bits_per_symbol());
        let sym = m.modulate(&bits, nsym);
        let e: f64 = sym[N_PREAMBLE..].iter().map(|s| s[0] * s[0] + s[1] * s[1]).sum::<f64>() / nsym as f64;
        assert!((e - 1.0).abs() < 0.03, "{}: 平均エネルギー {}", name, e);
    }
}

#[test]
fn llr_magnitude_follows_the_noise() {
    // 雑音が大きいほど、LLR の絶対値の平均は小さくなる（正しい向きの LLR が多い）
    let mut rng = Rng::new(3);
    let m = Modem::new(true, 5);
    let nsym = 4000;
    let bits = rng.bits(nsym * m.bits_per_symbol());
    let clean = m.modulate(&bits, nsym);
    let mut mean_abs = Vec::new();
    for std in [0.05, 0.2] {
        let noisy: Vec<[f64; 2]> = clean[N_PREAMBLE..]
            .iter()
            .map(|s| [s[0] + std * (rng.uniform() - 0.5) * 3.46, s[1] + std * (rng.uniform() - 0.5) * 3.46])
            .collect();
        let (llr, hard, _) = m.demodulate(&noisy, bits.len());
        mean_abs.push(llr.iter().map(|l| l.abs()).sum::<f64>() / llr.len() as f64);
        let errors = hard.iter().zip(&bits).filter(|(a, b)| a != b).count();
        assert!(errors < bits.len() / 3, "誤りが多すぎる: {}", errors);
    }
    assert!(mean_abs[0] > mean_abs[1]);
}

#[test]
fn scrambler_is_an_involution() {
    use gfm_core::scrambler::Scrambler;
    let mut rng = Rng::new(4);
    let s = Scrambler::new(500);
    let bits = rng.bits(500);
    let mut x = bits.clone();
    s.scramble(&mut x);
    assert_ne!(x, bits);
    // 受信側: LLR と硬判定のスクランブルを戻すと、元のビットになる
    let mut llr: Vec<f64> = x.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
    let mut hard = x.clone();
    s.descramble(&mut llr, &mut hard);
    assert_eq!(hard, bits);
    assert!(llr.iter().zip(&bits).all(|(&l, &b)| (l > 0.0) == (b == 0)));
}
