//! 欠落補間（PLC）。LPC 外挿。GraceFM `gracefm/plc.py` の移植。すべて整数演算。

use crate::dsp::{bit_length, saturate_bits};
use crate::tables::HANN128_Q15;

const PCM_BITS: u32 = 24;

/// x: 整数の履歴 → LPC 係数 a[0..=order]（Q15、a[0] = 1<<15）
pub fn lpc(x: &[i64], order: usize) -> Vec<i64> {
    let n = x.len();
    let denom = (n.max(2) - 1) as i64;
    let w: Vec<i64> = x.iter().enumerate().map(|(i, &v)| (v * HANN128_Q15[((i as i64 * 127) / denom) as usize]) >> 15).collect();
    let maxabs = w.iter().map(|v| v.abs()).max().unwrap_or(0);
    let shift = (bit_length(maxabs) - 15).max(0);
    let w: Vec<i64> = w.iter().map(|&v| v >> shift).collect();
    let mut r: Vec<i128> = (0..=order).map(|k| (0..n.saturating_sub(k)).map(|i| (w[i] * w[i + k]) as i128).sum()).collect();
    let mut a = vec![0i64; order + 1];
    a[0] = 1 << 15;
    if r[0] <= 0 {
        return a;
    }
    r[0] += r[0] >> 13;
    let mut err: i128 = r[0];
    for i in 1..=order {
        let mut acc: i128 = r[i] * (1 << 15);
        for j in 1..i {
            acc += a[j] as i128 * r[i - j];
        }
        let q = acc.abs() / err;
        let k = (if acc >= 0 { -q } else { q }).clamp(-(1 << 15) + 1, (1 << 15) - 1) as i64;
        let old = a.clone();
        for j in 1..i {
            a[j] = old[j] + ((k * old[i - j]) >> 15);
        }
        a[i] = k;
        err = (err * ((1i128 << 15) - ((k * k) >> 15) as i128)) >> 15;
        if err <= 0 {
            break;
        }
    }
    let mut bw: i64 = 1 << 15;
    for i in 1..=order {
        bw = (bw * 32702) >> 15;
        a[i] = (a[i] * bw) >> 15;
    }
    a
}

pub fn extrapolate(hist: &[i64], n: usize, order: usize) -> Vec<i64> {
    if hist.len() <= order {
        return vec![0; n];
    }
    let a = lpc(hist, order);
    let mut buf: Vec<i64> = hist[hist.len() - order..].to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut acc = 0i64;
        for j in 0..order {
            acc += a[1 + j] * buf[buf.len() - 1 - j];
        }
        let v = saturate_bits(-(acc >> 15), PCM_BITS);
        out.push(v);
        buf.push(v);
        buf.remove(0);
    }
    out
}

pub struct PlcParams {
    pub order: usize,
    pub hist_len: usize,
    pub hold_ms: usize,
    pub fade_ms: usize,
    pub xfade: usize,
}

impl Default for PlcParams {
    fn default() -> Self {
        PlcParams { order: 16, hist_len: 128, hold_ms: 10, fade_ms: 20, xfade: 16 }
    }
}

/// y: (n) の [L, R]、valid: (n) → 欠落区間を埋めた y
pub fn conceal(y: &[[i64; 2]], valid: &[bool], fs: usize, pp: &PlcParams) -> Vec<[i64; 2]> {
    let mut out: Vec<[i64; 2]> = y.to_vec();
    for (o, &v) in out.iter_mut().zip(valid) {
        if !v {
            *o = [0, 0];
        }
    }
    let hold = (fs * pp.hold_ms / 1000) as i64;
    let fade = ((fs * pp.fade_ms / 1000).max(1)) as i64;
    let n = y.len();
    let mut i = 0;
    while i < n {
        if valid[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !valid[j] {
            j += 1;
        }
        let l = j - i;
        let k = pp.xfade.min(n - j);
        for c in 0..2 {
            let hist: Vec<i64> = out[i.saturating_sub(pp.hist_len)..i].iter().map(|v| v[c]).collect();
            let ext = extrapolate(&hist, l + k, pp.order);
            let pred: Vec<i64> = ext
                .iter()
                .enumerate()
                .map(|(t, &e)| {
                    let g = ((1i64 << 15) - (((t as i64 - hold) << 15).div_euclid(fade))).clamp(0, 1 << 15);
                    (e * g) >> 15
                })
                .collect();
            for t in 0..l {
                out[i + t][c] = pred[t];
            }
            for m in 0..k {
                let w = ((m as i64 + 1) << 15) / (k as i64 + 1);
                out[j + m][c] = (w * out[j + m][c] + ((1 << 15) - w) * pred[l + m] + (1 << 14)) >> 15;
            }
        }
        i = j;
    }
    out
}

/// 情報のある（系統, 成分, 帯域）の組が変わったブロックの先頭 xfade サンプルを、直前の出力の LPC 外挿から新しい出力へクロスフェードでつなぐ
pub fn crossfade_switch(y: &[[i64; 2]], patterns: &[Vec<bool>], block: usize, pp: &PlcParams) -> Vec<[i64; 2]> {
    let mut out: Vec<[i64; 2]> = y.to_vec();
    let nb = patterns.len().min(y.len() / block);
    for i in 1..nb {
        if patterns[i] != patterns[i - 1] && patterns[i].iter().any(|&b| b) && patterns[i - 1].iter().any(|&b| b) {
            let s = i * block;
            let k = pp.xfade.min(y.len() - s);
            for c in 0..2 {
                let hist: Vec<i64> = out[s.saturating_sub(pp.hist_len)..s].iter().map(|v| v[c]).collect();
                let pred = extrapolate(&hist, k, pp.order);
                for m in 0..k {
                    let w = ((m as i64 + 1) << 15) / (k as i64 + 1);
                    out[s + m][c] = (w * out[s + m][c] + ((1 << 15) - w) * pred[m] + (1 << 14)) >> 15;
                }
            }
        }
    }
    out
}
