//! 試験用の補助: 再現できる擬似乱数と、試験信号。

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    /// [0, 1)
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    #[allow(dead_code)]
    pub fn bit(&mut self) -> u8 {
        (self.uniform() < 0.5) as u8
    }
    #[allow(dead_code)]
    pub fn bits(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.bit()).collect()
    }
}

/// 低音の正弦波 + 3 kHz の正弦波 + 小さな雑音のステレオ（32 kHz、満点 ±1）
#[allow(dead_code)]
pub fn stereo(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut rng = Rng::new(seed);
    let tau = 2.0 * std::f64::consts::PI;
    (0..n)
        .map(|i| {
            let t = i as f64 / 32000.0;
            let a = 0.3 * (tau * 220.0 * t).sin() + 0.1 * (tau * 3300.0 * t).sin() + 0.02 * (rng.uniform() - 0.5);
            let b = 0.25 * (tau * 330.0 * t).sin() + 0.02 * (rng.uniform() - 0.5);
            [a, b]
        })
        .collect()
}

/// 全プロファイル（Y 方式を除く）
#[allow(dead_code)]
pub const PROFILES: [&str; 7] = ["B200", "B200M", "M700", "M700M", "SUB100", "SUB50M", "SUB100-48K"];
