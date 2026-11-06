//! 超解像の判定: 元の音質が良いときだけ PostNet をかける。MendNet の事後標準偏差から出す予測の SNR
//! （過去 WIN フレームの、係数が使えたフレームの e0 の波形でのエネルギー / sd の同じもの。係数は直交ではないので、基底のエネルギーで重みを付ける）が ON 以上でオンになり、オンの間は OFF を下回るまでオン（ヒステリシス）。
//! 過去が WIN フレーム揃うまではオフ。

use std::collections::VecDeque;

use crate::codec::coef_power;
use crate::profiles::Profile;
use crate::restore::{Restored, NC};

/// 判定の窓 [フレーム]（1 秒）
pub const WIN: usize = 500;
/// オンにする予測の SNR [dB]
pub const ON: f64 = 27.5;
/// オンのあと、オフに戻す予測の SNR [dB]
pub const OFF: f64 = 25.5;

pub struct Gate {
    /// 係数ごとの基底のエネルギー
    power: Vec<f64>,
    /// 直近 WIN フレームの (信号、分散) の二乗和
    hist: VecDeque<(f64, f64)>,
    sig: f64,
    var: f64,
    seen: usize,
    on: bool,
}

impl Gate {
    pub fn new(p: &Profile) -> Gate {
        Gate { power: coef_power(p), hist: VecDeque::with_capacity(WIN + 1), sig: 0.0, var: 0.0, seen: 0, on: false }
    }

    /// 次のフレームの判定
    pub fn push(&mut self, r: &Restored) -> bool {
        let sq = |x: &[f32]| x.iter().enumerate().map(|(i, &v)| (v as f64).powi(2) * self.power[i % NC]).sum::<f64>();
        let (s, v) = if r.usable { (sq(&r.e0), sq(&r.sd)) } else { (0.0, 0.0) };
        self.hist.push_back((s, v));
        self.sig += s;
        self.var += v;
        if self.hist.len() > WIN {
            let (s0, v0) = self.hist.pop_front().unwrap();
            self.sig -= s0;
            self.var -= v0;
        }
        self.seen += 1;
        if self.seen <= WIN {
            return false;
        }
        let snr = 10.0 * (self.sig.max(1e-30) / self.var.max(1e-30)).log10();
        self.on = self.sig > 0.0 && snr >= if self.on { OFF } else { ON };
        self.on
    }

    /// 直前の判定
    pub fn is_on(&self) -> bool {
        self.on
    }
}
