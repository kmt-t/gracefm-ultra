//! 送信と受信の対称性: ノイズなしなら、Decoder(Encoder(x)) は硬判定の decode_block(encode_block(x)) と一致する
//! （軟判定の dequantize は、確定した bit では硬判定と同じ値になる）。

use gfm_core::codec::{decode_block, encode_block};
use gfm_core::decoder::Decoder;
use gfm_core::encoder::Encoder;
use gfm_core::profiles::profile_by_name;

fn signal(n: usize, seed: u64) -> Vec<[i64; 2]> {
    let mut s = seed;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5
    };
    (0..n)
        .map(|i| {
            let t = i as f64 / 32000.0;
            let a = 0.3 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() + 0.1 * (2.0 * std::f64::consts::PI * 3300.0 * t).sin() + 0.02 * rnd();
            let b = 0.25 * (2.0 * std::f64::consts::PI * 330.0 * t).sin() + 0.02 * rnd();
            [(a * 8388607.0) as i64, (b * 8388607.0) as i64]
        })
        .collect()
}

#[test]
fn decoder_inverts_encoder() {
    for name in ["B200M", "B200", "M700M", "M700", "SUB100", "SUB50M", "SUB100-48K"] {
        let p = profile_by_name(name).unwrap();
        let x = signal(64 * 40, 7);
        let frames: Vec<_> = Encoder::new(&p, 0, 0).unwrap().run(&x).into_iter().map(Some).collect();
        let out = Decoder::new(&p).unwrap().run(&frames, x.len(), 0);
        assert!(out.valid.iter().all(|&v| v), "{}: 欠落が出た", name);
        assert!(out.quality.iter().all(|&q| q == 0 || q == 1), "{}: 品質コード", name);
        for i in 0..40 {
            let blk = &x[i * 64..(i + 1) * 64];
            let want = decode_block(&encode_block(blk, &p, true), &p);
            assert_eq!(&out.pcm[i * 64..(i + 1) * 64], &want[..], "{}: ブロック {}", name, i);
        }
    }
}

#[test]
fn lost_frame_is_reported() {
    let p = profile_by_name("B200").unwrap();
    let x = signal(64 * 10, 3);
    let mut frames: Vec<_> = Encoder::new(&p, 0, 0).unwrap().run(&x).into_iter().map(Some).collect();
    frames[4] = None;
    let mut d = Decoder::new(&p).unwrap();
    let out = d.run(&frames, x.len(), 0);
    assert_eq!(out.quality[4], 2);
    assert!(!out.valid[4 * 64] && out.valid[3 * 64]);
    assert_eq!(d.stats.lost, 1);
}
