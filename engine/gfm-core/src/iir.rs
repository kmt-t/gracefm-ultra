//! 出力段の切り替え先の因果 IIR（Butterworth 4 次 4 kHz）と、品質コードによる切り替え。GraceFM `gracefm/iir.py`、`postfilter.select_by_quality` の移植。

use crate::dsp::saturate_bits;
use crate::tables::{ALT_IIR_32K_Q16, ALT_IIR_48K_Q16};

const Q: i64 = 16;
const PCM_BITS: u32 = 24;

/// 区間ごとの (b0_q, s, a1_q, a2_q)。b0 = b0_q / 2^(16+s)、a1 と a2 は Q16、a0 = 1、b1 = 2·b0、b2 = b0（係数はすべて 18 bit 符号つき）
pub fn alt_sections(fs: usize) -> &'static [[i64; 4]; 2] {
    if fs == 32000 { &ALT_IIR_32K_Q16 } else { &ALT_IIR_48K_Q16 }
}

/// 直接形 II の転置で区間を直列に通す（因果）。状態と出力は b0 の尺度（Q16+s）で持つ
pub fn iir_filter(x: &[[i64; 2]], sections: &[[i64; 4]]) -> Vec<[i64; 2]> {
    let mut st = vec![[[0i64; 2]; 2]; sections.len()]; // [区間][ch][s1, s2]
    let mut out = Vec::with_capacity(x.len());
    for xs in x {
        let mut o = [0i64; 2];
        for c in 0..2 {
            let mut v = xs[c];
            for (j, sec) in sections.iter().enumerate() {
                let (b0, sc, a1, a2) = (sec[0], sec[1], sec[2], sec[3]);
                let qs = Q + sc;
                let s = &mut st[j][c];
                let acc = b0 * v + s[0];
                let y = saturate_bits((acc + (1 << (qs - 1))) >> qs, PCM_BITS);
                s[0] = 2 * b0 * v - ((a1 * y) << sc) + s[1];
                s[1] = b0 * v - ((a2 * y) << sc);
                v = y;
            }
            o[c] = v;
        }
        out.push(o);
    }
    out
}

/// ブロックごとの品質コード（0 正常, 1 高域欠落, 2 欠落）に応じて、y と alt を切り替える。切り替えは先頭から fade サンプルの直線
pub fn select_by_quality(y: &[[i64; 2]], alt: &[[i64; 2]], quality: &[i64], block: usize, fade: i64) -> Vec<[i64; 2]> {
    let n = y.len();
    let mut out = y.to_vec();
    let mut w: i64 = 0;
    for (i, &q) in quality.iter().enumerate() {
        let s0 = i * block;
        if s0 >= n {
            break;
        }
        let s1 = n.min(s0 + block);
        let target: i64 = if q == 1 || q == 2 { 32768 } else { 0 };
        let mut r = w;
        for t in 0..s1 - s0 {
            let m = ((t + 1) as i64).min(fade);
            r = w + ((target - w) * m).div_euclid(fade);
            for c in 0..2 {
                out[s0 + t][c] = (y[s0 + t][c] * (32768 - r) + alt[s0 + t][c] * r + 16384) >> 15;
            }
        }
        w = r;
    }
    out
}

pub fn alt_delay(fs: usize) -> usize {
    if fs == 32000 { 3 } else { 5 }
}

/// 受信の出力 y に、欠落・高域欠落のブロックだけ IIR の出力を切り替える。出力は alt_delay サンプル遅れる
pub fn apply_alt(y: &[[i64; 2]], quality: &[i64], fs: usize, block: usize) -> Vec<[i64; 2]> {
    let d = alt_delay(fs);
    let alt = iir_filter(y, alt_sections(fs));
    let mut delayed = vec![[0i64; 2]; d];
    delayed.extend_from_slice(&y[..y.len() - d]);
    select_by_quality(&delayed, &alt, quality, block, (fs / 1000) as i64)
}
