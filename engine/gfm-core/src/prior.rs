//! ヘッダの事前知識つき復号（段 A）。ヘッダの情報ビットの多くは予測できるので、予測を事前の軟値として、ビタビ復号の情報ビット側に足す。
//!
//! 予測できるもの:
//! - プロファイル ID、形式識別子、暗号化ビット: 既知
//! - フレーム番号（16 bit）: 直前に復号できたフレームの番号 + 経過フレーム数
//! - 指数（4 bit）: 直前に復号できたフレームの指数（近いほど確か）、副系がある場合は、副系で復号した同じブロックの係数のピーク
//!
//! 事前の軟値が誤っていても、CRC が合わなければ採用されないので、誤ったヘッダを採ることはない（CRC の誤採用の確率は変わらない）。

use crate::codec::quant::{exponent_for_peak, BlockData};
use crate::layout::Layout;
use crate::profiles::MAX_BANDS;

/// 帯域ごとの係数のピーク（絶対値の最大、満点 = 1.0）: [成分][帯域]。NaN は不明
pub type Peaks = [[f64; MAX_BANDS]; 2];

/// 事前知識の設定
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorConfig {
    /// フレーム番号と既知の固定ビット
    pub counter: bool,
    /// 直前のフレームの指数
    pub temporal: bool,
    /// 副系のピークからの指数
    pub sub: bool,
}

impl PriorConfig {
    pub const NONE: PriorConfig = PriorConfig { counter: false, temporal: false, sub: false };
    pub const TEMPORAL: PriorConfig = PriorConfig { counter: true, temporal: true, sub: false };
    pub const FULL: PriorConfig = PriorConfig { counter: true, temporal: true, sub: true };
}

/// 既知のビットの事前の軟値の大きさ
const KNOWN: f64 = 12.0;
/// フレーム番号の事前の軟値の大きさ（番号が飛んでいたら CRC が合わないだけなので、控えめに）
const COUNTER: f64 = 6.0;
/// 直前の復号から何フレームまで、番号・指数の予測を使うか
const MAX_AGE: usize = 64;
/// 副系のピークからの指数の予測の確率: 当たり、±1
const SUB_HIT: f64 = 0.8;
const PROB_FLOOR: f64 = 1e-3;
const LLR_CLAMP: f64 = 12.0;

/// 直前に復号できたフレームの情報
#[derive(Default)]
pub struct Tracker {
    last: Option<(usize, u64)>,           // (フレームの位置, フレーム番号)
    exps: Vec<Vec<Option<(usize, i64)>>>, // [ch][k] → (フレームの位置, 指数インデックス)
}

impl Tracker {
    pub fn new() -> Tracker {
        Tracker::default()
    }

    /// 位置 idx のフレームが復号できた（基本 CRC が通った）
    pub fn observe(&mut self, idx: usize, counter: u64, bd: &BlockData) {
        self.last = Some((idx, counter));
        if self.exps.is_empty() {
            self.exps = bd.eidx.iter().map(|v| vec![None; v.len()]).collect();
        }
        for (ch, v) in bd.eidx.iter().enumerate() {
            for (k, e) in v.iter().enumerate() {
                if let Some(e) = e {
                    self.exps[ch][k] = Some((idx, *e));
                }
            }
        }
    }

    /// 位置 idx のフレームのヘッダの情報ビット（n_hdata 個）の事前の軟値（正 = 0 らしい）。hint: 副系のピーク
    pub fn header_prior(&self, lay: &Layout, idx: usize, cfg: &PriorConfig, hint: Option<&Peaks>) -> Vec<f64> {
        let mut prior = vec![0.0; lay.n_hdata];
        let p = &lay.p;
        if cfg.counter {
            // プロファイル ID（4）、系統（1、不明）、暗号化（1、0）、形式識別子（2）
            set_bits(&mut prior[0..4], p.pid as u64, KNOWN);
            prior[5] = KNOWN;
            set_bits(&mut prior[6..8], p.format_id as u64, KNOWN);
            if let Some((i0, c0)) = self.last.filter(|&(i0, _)| idx >= i0 && idx - i0 <= MAX_AGE) {
                set_bits(&mut prior[8..lay.header_bits], (c0 + (idx - i0) as u64) & 0xFFFF, COUNTER);
            }
        }
        for (n, &(ch, k)) in lay.exp_items.iter().enumerate() {
            let mut dist: Option<Vec<f64>> = None;
            let levels = 1usize << p.exp_bits;
            if cfg.temporal {
                if let Some((i0, e0)) = self.exps.get(ch).and_then(|v| v[k]).filter(|&(i0, _)| idx >= i0 && idx - i0 <= MAX_AGE) {
                    let scale = (0.5 + 0.1 * (idx - i0) as f64).min(3.0);
                    dist = Some((0..levels).map(|e| (-((e as i64 - e0).abs() as f64) / scale).exp() + PROB_FLOOR).collect());
                }
            }
            if cfg.sub {
                if let Some(peak) = hint.map(|h| h[ch][k]).filter(|v| v.is_finite()) {
                    let hit = exponent_for_peak(peak, k, p.bits(ch)[k], p);
                    let sub_dist: Vec<f64> = (0..levels)
                        .map(|e| match (e as i64 - hit).abs() {
                            0 => SUB_HIT,
                            1 => (1.0 - SUB_HIT) / 2.0,
                            _ => PROB_FLOOR,
                        })
                        .collect();
                    dist = Some(match dist {
                        Some(d) => d.iter().zip(&sub_dist).map(|(a, b)| a * b).collect(),
                        None => sub_dist,
                    });
                }
            }
            if let Some(d) = dist {
                let start = lay.header_bits + n * p.exp_bits;
                bit_llrs(&d, &mut prior[start..start + p.exp_bits]);
            }
        }
        prior
    }
}

/// 値 v の下位 out.len() bit（MSB から）に、正しい向きの軟値 ±a を入れる
fn set_bits(out: &mut [f64], v: u64, a: f64) {
    let n = out.len();
    for (i, o) in out.iter_mut().enumerate() {
        *o = if (v >> (n - 1 - i)) & 1 == 0 { a } else { -a };
    }
}

/// 値の分布 dist（添字 = 値）から、MSB から並べたビットごとの軟値 ln(P(0) / P(1))
fn bit_llrs(dist: &[f64], out: &mut [f64]) {
    let n = out.len();
    let total: f64 = dist.iter().sum();
    for (i, o) in out.iter_mut().enumerate() {
        let p1: f64 = dist.iter().enumerate().filter(|(e, _)| (e >> (n - 1 - i)) & 1 == 1).map(|(_, p)| p).sum::<f64>() / total;
        *o = ((1.0 - p1) / p1).ln().clamp(-LLR_CLAMP, LLR_CLAMP);
    }
}
