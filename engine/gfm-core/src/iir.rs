//! 出力段の切り替え先の因果 IIR（Butterworth 4 次 4 kHz）と、品質コードによる切り替え。GraceFM `gracefm/iir.py`、`postfilter.select_by_quality` の移植。

use crate::tables::{ALT_IIR_32K_Q28, ALT_IIR_48K_Q28};

const Q: u32 = 28;
const HALF: i64 = 1 << (Q - 1);
const PCM_MAX: i64 = (1 << 23) - 1;
const PCM_MIN: i64 = -(1 << 23);

pub fn alt_sections(fs: usize) -> &'static [[i64; 5]; 2] {
    if fs == 32000 { &ALT_IIR_32K_Q28 } else { &ALT_IIR_48K_Q28 }
}

pub fn iir_filter(x: &[[i64; 2]], sections: &[[i64; 5]]) -> Vec<[i64; 2]> {
    let mut st = vec![[[0i64; 2]; 2]; sections.len()]; // [区間][ch][s1, s2]
    let mut out = Vec::with_capacity(x.len());
    for xs in x {
        let mut o = [0i64; 2];
        for c in 0..2 {
            let mut v = xs[c];
            for (j, sec) in sections.iter().enumerate() {
                let (b0, b1, b2, a1, a2) = (sec[0], sec[1], sec[2], sec[3], sec[4]);
                let s = &mut st[j][c];
                let acc = b0 * v + s[0];
                let y = ((acc + HALF) >> Q).clamp(PCM_MIN, PCM_MAX);
                s[0] = b1 * v - a1 * y + s[1];
                s[1] = b2 * v - a2 * y;
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
