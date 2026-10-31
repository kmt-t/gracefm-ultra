//! 出力 IIR（Butterworth 4 次、4 kHz）: 直流利得 1、インパルス応答が減衰する、カットオフで -3 dB。

use gfm_core::iir::{butter4_lowpass, iir_filter};

#[test]
fn butterworth_is_stable_and_has_unit_dc_gain() {
    for fs in [32000.0, 48000.0] {
        let sec = butter4_lowpass(4000.0, fs);
        let step = vec![[1.0, 1.0]; 4000];
        let y = iir_filter(&step, &sec);
        assert!((y[3999][0] - 1.0).abs() < 1e-6, "直流利得 {}", y[3999][0]);
        let mut imp = vec![[0.0, 0.0]; 4000];
        imp[0] = [1.0, 1.0];
        let h = iir_filter(&imp, &sec);
        assert!(h.iter().all(|v| v[0].is_finite()) && h[3999][0].abs() < 1e-9, "インパルス応答が減衰しない");
        // カットオフ周波数の正弦波の振幅は約 1/√2
        let w = 2.0 * std::f64::consts::PI * 4000.0 / fs;
        let x: Vec<[f64; 2]> = (0..4000).map(|i| [(w * i as f64).sin(), 0.0]).collect();
        let y = iir_filter(&x, &sec);
        let peak = y[2000..].iter().map(|v| v[0].abs()).fold(0.0, f64::max);
        assert!((peak - 0.7071).abs() < 0.02, "カットオフでの振幅 {}", peak);
    }
}
