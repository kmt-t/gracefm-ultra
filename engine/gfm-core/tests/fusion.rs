//! 2 系統の融合と、欠落補間。

mod common;
use gfm_core::decoder::StreamResult;
use gfm_core::fusion::*;
use gfm_core::plc::*;

fn stream(pcm: Vec<[f64; 2]>, valid: bool, var: f64) -> StreamResult {
    let n = pcm.len();
    StreamResult {
        pcm,
        valid: vec![valid; n],
        quality: vec![0; n / 64],
        rb: vec![[[var; 5]; 2]; n],
        peaks: Vec::new(),
        stats: Default::default(),
    }
}

#[test]
fn weights_follow_the_inverse_variance() {
    let inf = f64::INFINITY;
    assert_eq!(weight_a(inf, inf), 0.5);
    assert_eq!(weight_a(inf, 1.0), 0.0);
    assert_eq!(weight_a(1.0, inf), 1.0);
    assert!((weight_a(2.0, 2.0) - 0.5).abs() < 1e-15);
    assert!((weight_a(1.0, 3.0) - 0.75).abs() < 1e-15, "分散が小さい A の重みが大きい");
}

#[test]
fn fusing_identical_streams_changes_nothing() {
    let x = common::stereo(64 * 6, 1);
    let (a, b) = (stream(x.clone(), true, 1e-6), stream(x.clone(), true, 4e-6));
    let (y, valid) = fuse_band(&a, &b, 64, 4);
    assert!(valid.iter().all(|&v| v));
    for (p, q) in x.iter().zip(&y) {
        assert!((p[0] - q[0]).abs() < 1e-12 && (p[1] - q[1]).abs() < 1e-12);
    }
}

#[test]
fn a_stream_that_did_not_arrive_is_ignored() {
    let x = common::stereo(64 * 4, 2);
    let other = common::stereo(64 * 4, 3);
    let (y, valid) = fuse_band(&stream(other, false, f64::INFINITY), &stream(x.clone(), true, 1e-6), 64, 4);
    assert!(valid.iter().all(|&v| v));
    assert!(x.iter().zip(&y).all(|(p, q)| (p[0] - q[0]).abs() < 1e-12));
    let (_, valid) = fuse_band(&stream(x.clone(), false, 1.0), &stream(x, false, 1.0), 64, 4);
    assert!(valid.iter().all(|&v| !v), "どちらも届かなければ無効");
}

#[test]
fn lpc_extrapolates_a_sinusoid() {
    // 1 kHz の正弦波（履歴に 4 周期）。LPC（16 次、ハン窓、帯域拡張）の外挿は、純音でも完全には続かない
    let w = 2.0 * std::f64::consts::PI * 1000.0 / 32000.0;
    let hist: Vec<f64> = (0..128).map(|i| 0.5 * (w * i as f64).sin()).collect();
    let a = lpc(&hist, 16);
    assert_eq!(a.len(), 17);
    assert_eq!(a[0], 1.0);
    let ext = extrapolate(&hist, 32, 16);
    let err: f64 = ext.iter().enumerate().map(|(i, v)| (v - 0.5 * (w * (128 + i) as f64).sin()).powi(2)).sum::<f64>() / 32.0;
    // 上限 0.101 は、本家 Python（整数）の LPC 外挿の同じ入力での rms 誤差
    assert!(err.sqrt() < 0.101, "外挿の誤差 {}", err.sqrt());
}

#[test]
fn lpc_of_silence_is_neutral() {
    let a = lpc(&[0.0; 128], 16);
    assert_eq!(a[0], 1.0);
    assert!(a[1..].iter().all(|&v| v == 0.0));
    assert!(extrapolate(&[0.0; 128], 10, 16).iter().all(|&v| v == 0.0));
}

#[test]
fn concealment_fills_a_gap_with_a_plausible_continuation() {
    let w = 2.0 * std::f64::consts::PI * 1000.0 / 32000.0;
    let x: Vec<[f64; 2]> = (0..2000).map(|i| [0.4 * (w * i as f64).sin(); 2]).collect();
    let mut valid = vec![true; 2000];
    valid[1000..1032].fill(false);
    let mut damaged = x.clone();
    damaged[1000..1032].fill([0.0, 0.0]);
    let y = conceal(&damaged, &valid, 32000, &PlcParams::default());
    let err: f64 = (1000..1032).map(|i| (y[i][0] - x[i][0]).powi(2)).sum::<f64>() / 32.0;
    let sig: f64 = (1000..1032).map(|i| x[i][0].powi(2)).sum::<f64>() / 32.0;
    // 上限 0.158 は、本家 Python（整数）の補間の同じ入力での誤差比
    assert!(err < 0.158 * sig, "補間の誤差が大きい: {} / {}", err, sig);
    assert!(y[..1000].iter().zip(&x).all(|(a, b)| a == b), "届いた区間は変えない");
}
