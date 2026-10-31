//! 2 系統の融合（帯域ごとの逆分散重み、浮動小数点）。GraceFM `gracefm/fusion.py` に相当。
//!
//! 2 系統のブロックの格子は同じなので、ブロック（seg サンプル）の区間を Haar 変換すると、各帯域の係数が両系統で 1 対 1 に対応する。
//! 係数ごとに、各系統のデコーダが見積もった誤差分散 R で w_A = R_B / (R_A + R_B) と重みづけする（M と S は別々に融合する）。
//! R は帯域ごと・半ブロックごと。係数が 2 つ以上ある帯域は、前半の係数に前半の R、後半の係数に後半の R を使う
//! （Haar の詳細係数の添字は時間位置に対応する）。係数 1 つの最低域は、両半分の重みの平均。
//! 送っていない帯域は R = ∞（情報なし）で、その帯域は送っている方の系統だけで決まる。
//! 一方の系統のブロックが届いていないときは、届いている方だけを採る。区間の出力は、その区間の 2 系統の値だけから決まる（未来を使わない）。

use crate::codec::{haar_forward, haar_inverse};
use crate::decoder::StreamResult;

/// w_A = R_B / (R_A + R_B)。情報なし（INFINITY）は重み 0 / 1 / 1/2
pub fn weight_a(ra: f64, rb: f64) -> f64 {
    match (ra.is_infinite(), rb.is_infinite()) {
        (true, true) => 0.5,
        (true, false) => 0.0,
        (false, true) => 1.0,
        _ => rb / (ra + rb),
    }
}

/// ブロック blk の先頭サンプルが有効か（ブロックの中では有効フラグは一定）
fn block_valid(v: &[bool], blk: usize, seg: usize) -> bool {
    v.get(blk * seg).copied().unwrap_or(false)
}

/// ブロック blk の M（ci = 0）か S（ci = 1）。端のブロックで足りないサンプルは 0
fn ms_block(y: &[[f64; 2]], blk: usize, seg: usize, ci: usize) -> Vec<f64> {
    (0..seg)
        .map(|t| {
            let [l, r] = y.get(blk * seg + t).copied().unwrap_or([0.0, 0.0]);
            if ci == 0 {
                (l + r) * 0.5
            } else {
                (l - r) * 0.5
            }
        })
        .collect()
}

/// ブロック blk の、成分 ci・帯域 k の誤差分散（off: ブロック内の位置）。範囲外は情報なし
fn rb_at(s: &StreamResult, blk: usize, seg: usize, off: usize, ci: usize, k: usize) -> f64 {
    s.rb.get(blk * seg + off).map_or(f64::INFINITY, |r| r[ci][k])
}

/// 2 系統 a, b（時刻合わせ済み）を融合する。seg: ブロック長、levels: Haar の段数。返り値: (融合した音声, 有効フラグ)
pub fn fuse_band(a: &StreamResult, b: &StreamResult, seg: usize, levels: usize) -> (Vec<[f64; 2]>, Vec<bool>) {
    let n = a.pcm.len();
    let mut y = Vec::with_capacity(n);
    let mut valid = Vec::with_capacity(n);
    for blk in 0..n.div_ceil(seg) {
        let (va, vb) = (block_valid(&a.valid, blk, seg), block_valid(&b.valid, blk, seg));
        let override_w = |w: f64| match (va, vb) {
            (true, false) => 1.0,
            (false, true) => 0.0,
            _ => w,
        };
        let fused: Vec<Vec<f64>> = (0..2)
            .map(|ci| {
                let ba = haar_forward(&ms_block(&a.pcm, blk, seg, ci), levels);
                let bb = haar_forward(&ms_block(&b.pcm, blk, seg, ci), levels);
                let bands: Vec<Vec<f64>> = ba
                    .iter()
                    .zip(&bb)
                    .enumerate()
                    .map(|(k, (xa, xb))| {
                        let nn = xa.len();
                        let w1 = weight_a(rb_at(a, blk, seg, 0, ci, k), rb_at(b, blk, seg, 0, ci, k));
                        let w2 = weight_a(rb_at(a, blk, seg, seg / 2, ci, k), rb_at(b, blk, seg, seg / 2, ci, k));
                        (0..nn)
                            .map(|j| {
                                let w = override_w(if nn == 1 {
                                    (w1 + w2) * 0.5
                                } else if j < nn / 2 {
                                    w1
                                } else {
                                    w2
                                });
                                w * xa[j] + (1.0 - w) * xb[j]
                            })
                            .collect()
                    })
                    .collect();
                haar_inverse(&bands)
            })
            .collect();
        for (m, sd) in fused[0].iter().zip(&fused[1]).take(n - blk * seg) {
            let v = va || vb;
            valid.push(v);
            y.push(if v { [m + sd, m - sd] } else { [0.0, 0.0] });
        }
    }
    (y, valid)
}
