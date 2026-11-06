//! 基本 CRC が通らないフレームの手がかり（Hint）: 仮数の受信 LLR から、係数の相対的な形が残ること。

mod common;
use gfm_core::channel::{awgn_adc, Gauss};
use gfm_core::codec::quant::qmax_of;
use gfm_core::decoder::{Decoder, Status};
use gfm_core::encoder::Encoder;
use gfm_core::framing::UnpackOptions;
use gfm_core::profiles::profile_by_name;

#[test]
fn hint_of_a_failed_frame_correlates_with_the_true_mantissa() {
    let p = profile_by_name("B200-L").unwrap();
    let dec = Decoder::new(&p).unwrap();
    let mut enc = Encoder::new(&p).unwrap();
    let mut noise = Gauss::new(7);
    let frames = gfm_core::codec::analyze_stream(&common::stereo(64 * 400, 3), &p);
    let (mut dot, mut norm, mut failed) = (0.0, 0.0, 0);
    for bands in &frames {
        let (bd, _, sym) = enc.frame_parts(bands);
        let rx = awgn_adc(&sym, 5.0, &mut noise);
        let (d, _) = dec.decode_frame(Some(&rx), &UnpackOptions { list: 1, ..UnpackOptions::default() });
        assert_eq!(d.hint.is_some(), d.status == Status::CrcFail, "手がかりは基本 CRC が通らないフレームだけ");
        let Some(h) = d.hint else { continue };
        failed += 1;
        let k = 0; // 帯域 0（4 係数）の M
        let b = p.bits(0)[k];
        let m = bd.mant[0][k].as_ref().unwrap();
        for (&mj, &hj) in m.iter().zip(&h.h[0]).take(p.ncoef(k)) {
            assert!(hj.abs() < 1.5);
            let t = mj as f64 / qmax_of(b) as f64;
            dot += t * hj;
            norm += t * t;
        }
        assert!((0.0..=1.0).contains(&h.q[0][k]));
    }
    assert!(failed > 50, "CRC 不一致のフレームが足りない: {failed}");
    assert!(dot / norm > 0.3, "手がかりが真の仮数と相関しない: {}", dot / norm);
}
