//! 送信と受信の対称性: ノイズなしなら、Decoder(Encoder(x)) の係数を合成すると、原音を量子化誤差の範囲で再現する。欠落は Lost で報告される。

use gfm_core::codec::synthesize_stream;
use gfm_core::decoder::{DecodedFrame, Decoder, Status};
use gfm_core::encoder::Encoder;
use gfm_core::profiles::profile_by_name;

fn signal(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut s = seed;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5
    };
    (0..n)
        .map(|i| {
            let t = i as f64 / 32000.0;
            let a =
                0.3 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() + 0.1 * (2.0 * std::f64::consts::PI * 3300.0 * t).sin() + 0.02 * rnd();
            let b = 0.25 * (2.0 * std::f64::consts::PI * 330.0 * t).sin() + 0.02 * rnd();
            [a, b]
        })
        .collect()
}

fn decode(p: &gfm_core::profiles::Profile, frames: &[Option<Vec<[f64; 2]>>]) -> (Vec<DecodedFrame>, Decoder) {
    let mut dec = Decoder::new(p).unwrap();
    let out = frames.iter().map(|f| dec.push(f.as_deref())).collect();
    (out, dec)
}

fn frames_of(p: &gfm_core::profiles::Profile, x: &[[f64; 2]]) -> Vec<Option<Vec<[f64; 2]>>> {
    Encoder::new(p).unwrap().run(x).into_iter().map(Some).collect()
}

#[test]
fn decoder_inverts_encoder() {
    // 各プロファイルの量子化だけの SNR の下限（dB、実測より 2 dB 下）。S を送らないプロファイルは、ステレオの原音との SNR が低くなる
    for (name, min_snr) in [
        ("B200", 47.5),
        ("B200Y", 30.0),
        ("M700Y", 30.0),
        ("B200MY", 1.5),
        ("M700MY", 1.5),
        ("B200Y-L", 30.0),
        ("M700Y-L", 30.0),
        ("B200MY-L", 1.5),
        ("M700MY-L", 1.5),
        ("M700", 38.0),
        ("SUB100", 12.5),
        ("SUB100-48K", 10.5),
        ("B200M", 1.5),
        ("M700M", 1.5),
        ("SUB50M", 1.0),
        ("B200-L", 47.5),
        ("M700-L", 38.0),
        ("SUB100-L", 12.5),
        ("SUB100-48K-L", 10.5),
        ("B200M-L", 1.5),
        ("M700M-L", 1.5),
        ("SUB50M-L", 1.0),
    ] {
        let p = profile_by_name(name).unwrap();
        let x = signal(64 * 60, 7);
        let frames = frames_of(&p, &x);
        let (out, _) = decode(&p, &frames);
        assert!(out.iter().all(|d| d.status == Status::Ok), "{}: 正常でないフレームが出た", name);
        let coefs: Vec<_> = out.iter().map(|d| d.coef.clone()).collect();
        let pcm = synthesize_stream(&coefs, &p);
        let (mut sig, mut err) = (0f64, 0f64);
        for (o, r) in pcm.iter().zip(&x).skip(128) {
            for c in 0..2 {
                let rv = r[c];
                sig += rv * rv;
                err += (o[c] - rv) * (o[c] - rv);
            }
        }
        let snr = 10.0 * (sig / err).log10();
        assert!(snr > min_snr, "{}: SNR {:.1} dB が下限 {} dB 未満", name, snr, min_snr);
    }
}

#[test]
fn lost_frame_is_reported() {
    let p = profile_by_name("B200-L").unwrap();
    let x = signal(64 * 10, 3);
    let mut frames = frames_of(&p, &x);
    frames[4] = None;
    let (out, dec) = decode(&p, &frames);
    assert_eq!(out[4].status, Status::Lost);
    assert_eq!(out[3].status, Status::Ok);
    assert_eq!(dec.stats.lost, 1);
}
