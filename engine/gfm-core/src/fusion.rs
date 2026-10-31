//! 2 系統の融合（帯域ごとの逆分散重み）。GraceFM `gracefm/fusion.py` の移植。

use crate::codec::{haar_forward, haar_inverse, ms_to_lr};
use crate::dsp::inv1p2pow_q15;

pub const INF_LOG: i64 = 1 << 40;
const HALF: i64 = 1 << 14;
const ONE: i64 = 1 << 15;

/// w_A = 1 / (1 + 2^(LA - LB)) (Q15)。情報なし (INF_LOG) は重み 0 / 1 / 1/2
pub fn weight_a(la: i64, lb: i64) -> i64 {
    let d = if la >= INF_LOG || lb >= INF_LOG { 0 } else { (la - lb).clamp(-(16 << 8), 16 << 8) };
    let w = inv1p2pow_q15(d);
    if la >= INF_LOG {
        if lb >= INF_LOG { HALF } else { 0 }
    } else if lb >= INF_LOG {
        ONE
    } else {
        w
    }
}

/// rb[サンプル][ci][k]。y: (n) の [L, R]。返り値: (融合した y, valid)
#[allow(clippy::too_many_arguments)]
pub fn fuse_band(
    ya: &[[i64; 2]], va: &[bool], yb: &[[i64; 2]], vb: &[bool], seg: usize, levels: usize,
    rba: &[[[i64; 5]; 2]], rbb: &[[[i64; 5]; 2]],
) -> (Vec<[i64; 2]>, Vec<bool>) {
    let n = ya.len();
    let nb = levels + 1;
    let ns = (n + seg - 1) / seg;
    let h = seg / 2;
    let get = |y: &[[i64; 2]], blk: usize, ci: usize, t: usize| -> i64 {
        let i = blk * seg + t;
        let (l, r) = if i < y.len() { (y[i][0], y[i][1]) } else { (0, 0) };
        if ci == 0 { (l + r + 2) >> 2 } else { (l - r + 2) >> 2 }
    };
    let blk_valid = |v: &[bool], blk: usize| -> bool { blk * seg < v.len() && v[blk * seg] };
    let rb_at = |rb: &[[[i64; 5]; 2]], blk: usize, off: usize, ci: usize, k: usize| -> i64 {
        let i = blk * seg + off;
        if i < rb.len() { rb[i][ci][k] } else { INF_LOG }
    };
    let mut out_ms = vec![[vec![0i64; seg], vec![0i64; seg]]; ns];
    for blk in 0..ns {
        let (vab, vbb) = (blk_valid(va, blk), blk_valid(vb, blk));
        let (only_a, only_b) = (vab && !vbb, vbb && !vab);
        let ov = |w: i64| -> i64 { if only_a { ONE } else if only_b { 0 } else { w } };
        for ci in 0..2 {
            let xa: Vec<i64> = (0..seg).map(|t| get(ya, blk, ci, t)).collect();
            let xb: Vec<i64> = (0..seg).map(|t| get(yb, blk, ci, t)).collect();
            let ba = haar_forward(&xa, levels);
            let bb = haar_forward(&xb, levels);
            let mut bands = Vec::with_capacity(nb);
            for k in 0..nb {
                let nn = ba[k].len();
                let w1 = weight_a(rb_at(rba, blk, 0, ci, k), rb_at(rbb, blk, 0, ci, k));
                let w2 = weight_a(rb_at(rba, blk, h, ci, k), rb_at(rbb, blk, h, ci, k));
                let band: Vec<i64> = (0..nn)
                    .map(|j| {
                        let w = if nn == 1 { ov((w1 + w2) >> 1) } else if j < nn / 2 { ov(w1) } else { ov(w2) };
                        (w * ba[k][j] + (ONE - w) * bb[k][j] + HALF) >> 15
                    })
                    .collect();
                bands.push(band);
            }
            out_ms[blk][ci] = haar_inverse(&bands);
        }
    }
    let mut y = Vec::with_capacity(n);
    let mut valid = Vec::with_capacity(n);
    for i in 0..n {
        let blk = i / seg;
        let v = blk_valid(va, blk) || blk_valid(vb, blk);
        valid.push(v);
        let lr = ms_to_lr(out_ms[blk][0][i % seg], out_ms[blk][1][i % seg]);
        y.push(if v { lr } else { [0, 0] });
    }
    (y, valid)
}
