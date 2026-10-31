//! 出力 IIR（真の Butterworth、4 次、4 kHz）と、品質コードによる切り替え。GraceFM `gracefm/iir.py`、`postfilter.select_by_quality` に相当。

/// 4 次 Butterworth ローパス（カットオフ fc [Hz]）の 2 つの biquad（正規化済み、a0 = 1）: [b0, b1, b2, a1, a2]
pub fn butter4_lowpass(fc: f64, fs: f64) -> [[f64; 5]; 2] {
    let w0 = 2.0 * std::f64::consts::PI * fc / fs;
    let (cw, sw) = (w0.cos(), w0.sin());
    let q = [0.5 / (std::f64::consts::PI / 8.0).cos(), 0.5 / (3.0 * std::f64::consts::PI / 8.0).cos()];
    let mut out = [[0.0; 5]; 2];
    for (i, &qi) in q.iter().enumerate() {
        let alpha = sw / (2.0 * qi);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cw) / 2.0 / a0;
        out[i] = [b0, 2.0 * b0, b0, -2.0 * cw / a0, (1.0 - alpha) / a0];
    }
    out
}

/// 直接形 II の転置で biquad を直列に通す（因果）
pub fn iir_filter(x: &[[f64; 2]], sections: &[[f64; 5]]) -> Vec<[f64; 2]> {
    let mut st = vec![[[0f64; 2]; 2]; sections.len()];
    let mut out = Vec::with_capacity(x.len());
    for xs in x {
        let mut o = [0f64; 2];
        for c in 0..2 {
            let mut v = xs[c];
            for (j, sec) in sections.iter().enumerate() {
                let s = &mut st[j][c];
                let y = sec[0] * v + s[0];
                s[0] = sec[1] * v - sec[3] * y + s[1];
                s[1] = sec[2] * v - sec[4] * y;
                v = y;
            }
            o[c] = v;
        }
        out.push(o);
    }
    out
}

/// 品質コード（0 正常, 1 高域欠落, 2 欠落）に応じて y と alt を切り替える。切り替えは、そのブロックの先頭から fade サンプルの直線
pub fn select_by_quality(y: &[[f64; 2]], alt: &[[f64; 2]], quality: &[i64], block: usize, fade: usize) -> Vec<[f64; 2]> {
    let n = y.len();
    let mut out = y.to_vec();
    let mut w = 0.0f64;
    for (i, &q) in quality.iter().enumerate() {
        let s0 = i * block;
        if s0 >= n {
            break;
        }
        let s1 = n.min(s0 + block);
        let target = if q == 1 || q == 2 { 1.0 } else { 0.0 };
        let mut r = w;
        for t in 0..s1 - s0 {
            let m = ((t + 1) as f64).min(fade as f64);
            r = w + (target - w) * m / fade as f64;
            for c in 0..2 {
                out[s0 + t][c] = y[s0 + t][c] * (1.0 - r) + alt[s0 + t][c] * r;
            }
        }
        w = r;
    }
    out
}

/// 欠落・高域欠落のブロックだけ IIR の出力へ切り替える。出力は全体を alt_delay サンプル遅らせて揃える（整数版と同じ）
pub fn apply_alt(y: &[[f64; 2]], quality: &[i64], fs: usize, block: usize) -> Vec<[f64; 2]> {
    let d = alt_delay(fs);
    let alt = iir_filter(y, &butter4_lowpass(4000.0, fs as f64));
    let mut delayed = vec![[0f64; 2]; d];
    delayed.extend_from_slice(&y[..y.len() - d]);
    select_by_quality(&delayed, &alt, quality, block, fs / 1000)
}

pub fn alt_delay(fs: usize) -> usize {
    if fs == 32000 {
        3
    } else {
        5
    }
}
