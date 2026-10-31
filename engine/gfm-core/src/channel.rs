//! 評価・学習データ用の通信路モデル（アナログの通信路を模擬する。GraceFM `gracefm/channel.py` に相当）。
//! フレームごとの電力利得（AWGN / Rayleigh）、Gilbert-Elliott 型のフレーム欠落、AWGN、12 bit の ADC。

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

pub struct Gauss {
    rng: Pcg64,
    spare: Option<f64>,
}

impl Gauss {
    pub fn new(seed: u64) -> Gauss {
        Gauss { rng: Pcg64::seed_from_u64(seed), spare: None }
    }
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = 1.0 - self.rng.gen::<f64>();
        let u2 = self.rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
}

/// フレームごとの電力利得（平均 1）。rayleigh: 複素利得を AR(1)（時定数 50 フレーム）で相関させる。受信側は理想的な AGC とし、利得の分だけ Es/N0 が変わる
pub fn fading_gains(nfr: usize, rayleigh: bool, g: &mut Gauss) -> Vec<f64> {
    if !rayleigh {
        return vec![1.0; nfr];
    }
    let rho = (-1.0f64 / 50.0).exp();
    let s = (1.0 - rho * rho).sqrt();
    let cn = |g: &mut Gauss| (g.normal() / 2f64.sqrt(), g.normal() / 2f64.sqrt());
    let mut h = cn(g);
    let mut out = vec![h.0 * h.0 + h.1 * h.1];
    for _ in 1..nfr {
        let w = cn(g);
        h = (rho * h.0 + s * w.0, rho * h.1 + s * w.1);
        out.push(h.0 * h.0 + h.1 * h.1);
    }
    out
}

/// Gilbert-Elliott 型の欠落。rate: 欠落率、burst: 平均バースト長 [フレーム]
pub fn loss_mask(nfr: usize, rate: f64, burst: f64, rng: &mut Pcg64) -> Vec<bool> {
    if rate <= 0.0 {
        return vec![false; nfr];
    }
    let p_bg = 1.0 / burst;
    let p_gb = rate * p_bg / (1.0 - rate);
    let mut bad = false;
    let mut out = Vec::with_capacity(nfr);
    for _ in 0..nfr {
        out.push(bad);
        bad = if bad { rng.gen::<f64>() >= p_bg } else { rng.gen::<f64>() < p_gb };
    }
    out
}

pub const ADC_BITS: i64 = 12;
pub const ADC_ONE: f64 = (1 << (ADC_BITS - 2)) as f64;

/// シンボルに Es/N0 = snr_db の複素白色雑音を加え、12 bit の ADC で量子化する（±1 = 1.0 の実数）
pub fn awgn_adc(sym: &[[f64; 2]], snr_db: f64, g: &mut Gauss) -> Vec<[f64; 2]> {
    let n0 = 10f64.powf(-snr_db / 10.0);
    let std = (n0 / 2.0).sqrt();
    sym.iter()
        .map(|s| {
            let mut o = [0f64; 2];
            for ax in 0..2 {
                let v = s[ax] + std * g.normal();
                let code = (v * ADC_ONE).round_ties_even().clamp(-(1 << (ADC_BITS - 1)) as f64, ((1 << (ADC_BITS - 1)) - 1) as f64);
                o[ax] = code / ADC_ONE;
            }
            o
        })
        .collect()
}

/// 通信路のシナリオ: 基準 Es/N0 [dB]、Rayleigh かどうか、欠落率、平均バースト長
#[derive(Clone, Copy, Debug)]
pub struct Scen {
    pub e0: f64,
    pub fade: bool,
    pub rate: f64,
    pub burst: f64,
}

/// 1 つのシナリオの、フレームごとの通信路の状態（利得込みの Es/N0、欠落）
pub struct Link {
    pub esn0: Vec<f64>,
    pub lost: Vec<bool>,
    pub noise: Gauss,
}

impl Link {
    /// 乱数の種は seed·1000 + si を基準にする（フェージング、欠落、雑音で別の系列）
    pub fn new(sc: &Scen, nfr: usize, seed: u64, si: u64) -> Link {
        let mut gfade = Gauss::new(seed * 1000 + si);
        let gains = fading_gains(nfr, sc.fade, &mut gfade);
        let mut lrng = Pcg64::seed_from_u64(seed * 1000 + 500 + si);
        let lost = loss_mask(nfr, sc.rate, sc.burst, &mut lrng);
        let esn0 = gains.iter().map(|g| sc.e0 + 10.0 * g.max(1e-6).log10()).collect();
        Link { esn0, lost, noise: Gauss::new(seed * 1000 + 900 + si) }
    }
}
