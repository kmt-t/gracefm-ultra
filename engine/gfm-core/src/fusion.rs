//! 2 系統の融合（帯域ごとの逆分散重み、浮動小数点）。GraceFM `gracefm/fusion.py` に相当。
//! ブロックの区間を Haar 変換すると、各帯域の係数が両系統で 1 対 1 に対応する。係数ごとに、各系統の誤差分散 R で w_A = R_B / (R_A + R_B) と重みづけする。

use crate::codec::quant::Rb;
use crate::codec::{haar_forward_f, haar_inverse};

/// w_A = R_B / (R_A + R_B)。情報なし（INFINITY）は重み 0 / 1 / 1/2
pub fn weight_a(ra: f64, rb: f64) -> f64 {
    match (ra.is_infinite(), rb.is_infinite()) {
        (true, true) => 0.5,
        (true, false) => 0.0,
        (false, true) => 1.0,
        _ => rb / (ra + rb),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fuse_band(
    ya: &[[f64; 2]], va: &[bool], yb: &[[f64; 2]], vb: &[bool], seg: usize, levels: usize, rba: &[Rb], rbb: &[Rb],
) -> (Vec<[f64; 2]>, Vec<bool>) {
    let n = ya.len();
    let nb = levels + 1;
    let ns = (n + seg - 1) / seg;
    let h = seg / 2;
    let ms = |y: &[[f64; 2]], blk: usize, ci: usize, t: usize| -> f64 {
        let i = blk * seg + t;
        let (l, r) = if i < y.len() { (y[i][0], y[i][1]) } else { (0.0, 0.0) };
        if ci == 0 { (l + r) * 0.5 } else { (l - r) * 0.5 }
    };
    let blk_valid = |v: &[bool], blk: usize| -> bool { blk * seg < v.len() && v[blk * seg] };
    let rb_at = |rb: &[Rb], blk: usize, off: usize, ci: usize, k: usize| -> f64 {
        let i = blk * seg + off;
        if i < rb.len() { rb[i][ci][k] } else { f64::INFINITY }
    };
    let mut out_ms = vec![[vec![0f64; seg], vec![0f64; seg]]; ns];
    for blk in 0..ns {
        let (vab, vbb) = (blk_valid(va, blk), blk_valid(vb, blk));
        let (only_a, only_b) = (vab && !vbb, vbb && !vab);
        let ov = |w: f64| -> f64 { if only_a { 1.0 } else if only_b { 0.0 } else { w } };
        for ci in 0..2 {
            let xa: Vec<f64> = (0..seg).map(|t| ms(ya, blk, ci, t)).collect();
            let xb: Vec<f64> = (0..seg).map(|t| ms(yb, blk, ci, t)).collect();
            let ba = haar_forward_f(&xa, levels);
            let bb = haar_forward_f(&xb, levels);
            let mut bands = Vec::with_capacity(nb);
            for k in 0..nb {
                let nn = ba[k].len();
                let w1 = weight_a(rb_at(rba, blk, 0, ci, k), rb_at(rbb, blk, 0, ci, k));
                let w2 = weight_a(rb_at(rba, blk, h, ci, k), rb_at(rbb, blk, h, ci, k));
                let band: Vec<f64> = (0..nn)
                    .map(|j| {
                        let w = if nn == 1 { ov((w1 + w2) * 0.5) } else if j < nn / 2 { ov(w1) } else { ov(w2) };
                        w * ba[k][j] + (1.0 - w) * bb[k][j]
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
        let (m, s) = (out_ms[blk][0][i % seg], out_ms[blk][1][i % seg]);
        y.push(if v { [m + s, m - s] } else { [0.0, 0.0] });
    }
    (y, valid)
}

