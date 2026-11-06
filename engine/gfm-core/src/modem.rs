//! 変調（送信）と復調（受信）。GraceFM `gracefm/modem.py` に相当。QPSK、階層 16QAM、Y 方式（QASK）。浮動小数点（信号点の平均エネルギーは 1）。

pub const N_PREAMBLE: usize = 16;

/// スクランブル系列: 1 + x^14 + x^15、初期値 100101010000000
pub fn prbs(n: usize) -> Vec<u8> {
    let mut reg: [u8; 15] = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let b = reg[13] ^ reg[14];
        out.push(b);
        for i in (1..15).rev() {
            reg[i] = reg[i - 1];
        }
        reg[0] = b;
    }
    out
}

/// Y 方式の半径の段のビット数（仮数 7 bit の最上位は位相、残りが半径）と、隣の段の間隔（登録プロファイルは全部同じ）
pub const Y_LOWER: usize = 6;
const Y_LEVELS: usize = 1 << Y_LOWER;
const Y_STEP: f64 = 2.0 / 105.0;

fn y_scale() -> f64 {
    let l = Y_LEVELS as f64;
    (1.0 + Y_STEP * Y_STEP * (l * l - 1.0) / 12.0).sqrt()
}

fn y_radius(level: f64) -> f64 {
    (1.0 + Y_STEP * (level - (Y_LEVELS - 1) as f64 / 2.0)) / y_scale()
}

/// Y の信号点の象限の符号: パリティが I 軸、最上位ビットが Q 軸
fn y_sign(msb: u8, parity: u8) -> [f64; 2] {
    [1.0 - 2.0 * parity as f64, 1.0 - 2.0 * msb as f64]
}

/// Y 方式の変調。u: 最上位ビット·2^Y_LOWER + 半径の段、parity: 最上位ビットのパリティ、mask: 係数ごとに 3 ビットのスクランブル
pub fn y_modulate(u: &[i64], parity: &[u8], mask: &[u8]) -> Vec<[f64; 2]> {
    let levels = Y_LEVELS as i64;
    (0..u.len())
        .map(|i| {
            let (m, l) = (mask[3 * i..3 * i + 3][0], u[i] % levels);
            let l = if mask[3 * i + 2] != 0 { levels - 1 - l } else { l };
            let r = y_radius(l as f64) * std::f64::consts::FRAC_1_SQRT_2;
            let s = y_sign((u[i] / levels) as u8 ^ m, parity[i] ^ mask[3 * i + 1]);
            [s[0] * r, s[1] * r]
        })
        .collect()
}

/// 専用シンボルの位相から、最上位ビットとパリティの LLR（自然対数、正 = 0 らしい）。var: 軸あたりの雑音分散
pub fn y_phase_llr(sym: &[[f64; 2]], mask: &[u8], var: f64) -> (Vec<f64>, Vec<f64>) {
    let (lo, hi) = (y_radius(0.0), y_radius((Y_LEVELS - 1) as f64));
    let (mut msb, mut par) = (Vec::with_capacity(sym.len()), Vec::with_capacity(sym.len()));
    for (i, y) in sym.iter().enumerate() {
        let unit = y[0].hypot(y[1]).clamp(lo, hi) * std::f64::consts::FRAC_1_SQRT_2;
        let flip = |b: u8| if b != 0 { -1.0 } else { 1.0 };
        msb.push((2.0 * y[1] * unit / var).clamp(-1000.0, 1000.0) * flip(mask[3 * i]));
        par.push((2.0 * y[0] * unit / var).clamp(-1000.0, 1000.0) * flip(mask[3 * i + 1]));
    }
    (msb, par)
}

/// 象限（最上位ビットとパリティ）を決めた受信点の、半径の段の観測値と、段の単位の雑音分散
pub fn y_level(y: &[f64; 2], msb: u8, parity: u8, mask: &[u8], var: f64) -> (f64, f64) {
    let s = y_sign(msb ^ mask[0], parity ^ mask[1]);
    let proj = (s[0] * y[0] + s[1] * y[1]) * std::f64::consts::FRAC_1_SQRT_2;
    let scale = y_scale();
    let top = (Y_LEVELS - 1) as f64;
    let obs = (proj * scale - 1.0) / Y_STEP + top / 2.0;
    (if mask[2] != 0 { top - obs } else { obs }, var * scale * scale / (Y_STEP * Y_STEP))
}

/// 半径の段の観測値 → 推定値と誤差分散（段の一様な事前分布へ向けて縮める）
pub fn y_estimate(obs: f64, noise: f64) -> (f64, f64) {
    let l = Y_LEVELS as f64;
    let prior = (l * l - 1.0) / 12.0;
    let gain = prior / (prior + noise);
    let center = (l - 1.0) / 2.0;
    ((center + gain * (obs - center)).clamp(0.0, l - 1.0), gain * noise)
}

/// 半径の段の各ビットの LLR（max-log、上位から Y_LOWER 個、正 = 0 らしい）
pub fn y_level_llr(obs: f64, noise: f64) -> [f64; Y_LOWER] {
    let mut out = [0.0; Y_LOWER];
    for (i, x) in out.iter_mut().enumerate() {
        let bit = Y_LOWER - 1 - i;
        let (mut d1, mut d0) = (f64::MAX, f64::MAX);
        for level in 0..Y_LEVELS {
            let d = (obs - level as f64) * (obs - level as f64);
            if (level >> bit) & 1 == 1 {
                d1 = d1.min(d);
            } else {
                d0 = d0.min(d);
            }
        }
        *x = ((d1 - d0) / (2.0 * noise)).clamp(-1000.0, 1000.0);
    }
    out
}

pub struct Modem {
    pub qam16h: bool,
    pub nlp: usize,
    pub mags: Vec<f64>,
    pub g2i: Vec<usize>,
    pub vals: Vec<f64>,
    pub lab: Vec<Vec<u8>>,
    pub preamble: Vec<[f64; 2]>,
}

impl Modem {
    pub fn new(qam16h: bool, alpha: i64) -> Modem {
        let l: usize = if qam16h { 2 } else { 1 };
        let a = if l > 1 { alpha } else { 1 };
        let nlp = l.trailing_zeros() as usize;
        let pos: Vec<f64> = (0..l as i64).map(|i| (a + 2 * i) as f64).collect();
        let r = (2.0 * pos.iter().map(|p| p * p).sum::<f64>() / l as f64).sqrt(); // 平均シンボルエネルギーが 1 になる規格化
        let mags: Vec<f64> = pos.iter().map(|p| p / r).collect();
        let mut vals = Vec::new();
        let mut lab = Vec::new();
        for hp in 0..2i64 {
            for (idx, &mag) in mags.iter().enumerate() {
                let g = idx ^ (idx >> 1);
                vals.push((1 - 2 * hp) as f64 * mag);
                let mut lb = vec![hp as u8];
                for j in 0..nlp {
                    lb.push(((g >> (nlp - 1 - j)) & 1) as u8);
                }
                lab.push(lb);
            }
        }
        let mut g2i = vec![0usize; l];
        for i in 0..l {
            g2i[i ^ (i >> 1)] = i;
        }
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let pb: [f64; 16] = [1., 1., 1., -1., -1., 1., 1., -1., 1., -1., -1., -1., 1., 1., -1., 1.];
        let preamble = (0..16).map(|i| [pb[i] * inv_sqrt2, pb[(i + 16 - 5) % 16] * inv_sqrt2]).collect();
        Modem { qam16h, nlp, mags, g2i, vals, lab, preamble }
    }

    pub fn bits_per_symbol(&self) -> usize {
        2 * (self.nlp + 1)
    }

    /// 1 フレームのシンボル数: max(ceil(n_tx / bps), ceil(n_coded / 2))
    pub fn n_symbols(&self, n_tx: usize, n_coded: usize) -> usize {
        let bps = self.bits_per_symbol();
        n_tx.div_ceil(bps).max(n_coded.div_ceil(2))
    }

    /// ビット列（nsym * bps 個）→ プリアンブル付きのシンボル
    pub fn modulate(&self, bits: &[u8], nsym: usize) -> Vec<[f64; 2]> {
        let nslot = 2 * nsym;
        let mut buf = vec![0u8; nslot * (1 + self.nlp)];
        buf[..bits.len()].copy_from_slice(bits);
        let mut v = vec![0f64; nslot];
        for s in 0..nslot {
            let mut code = 0usize;
            for j in 0..self.nlp {
                code = (code << 1) | buf[nslot * (1 + j) + s] as usize;
            }
            let mag = self.mags[self.g2i[code]];
            v[s] = if buf[s] == 1 { -mag } else { mag };
        }
        let mut out = self.preamble.clone();
        for n in 0..nsym {
            out.push([v[2 * n], v[2 * n + 1]]);
        }
        out
    }

    /// 復調（浮動小数点）。sym: プリアンブルを除いた受信シンボル（±1 = 1.0 の実数）。
    /// 返り値: (LLR（自然対数、正 = 0 らしい、先頭 n_bits 個）, 硬判定, 軸あたりの雑音分散)。LLR = (D1 - D0) / (2σ²)、σ² は最寄りの信号点との差から
    pub fn demodulate(&self, sym: &[[f64; 2]], n_bits: usize) -> (Vec<f64>, Vec<u8>, f64) {
        let nsym = sym.len();
        let nlp = self.nlp;
        let vals = &self.vals;
        let nv = vals.len();
        let mut total = 0f64;
        for y in sym {
            for &v in y {
                let mut bd = f64::MAX;
                for &c in vals {
                    bd = bd.min((v - c) * (v - c));
                }
                total += bd;
            }
        }
        let var = (total / (2 * nsym).max(1) as f64).max(1e-12);
        let nslot = 2 * nsym;
        let mut llr = vec![0f64; nslot * (1 + nlp)];
        let mut hard = vec![0u8; nslot * (1 + nlp)];
        let mut dist = vec![0f64; nv];
        for (n, y) in sym.iter().enumerate() {
            for ax in 0..2 {
                for (d, &c) in dist.iter_mut().zip(vals) {
                    *d = (y[ax] - c) * (y[ax] - c);
                }
                let best = dist.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0); // 同じ距離なら先頭
                for level in 0..=nlp {
                    hard[nslot * level + 2 * n + ax] = self.lab[best][level];
                    let (mut d1, mut d0) = (f64::MAX, f64::MAX);
                    for (lab, &d) in self.lab.iter().zip(&dist) {
                        if lab[level] == 1 {
                            d1 = d1.min(d);
                        } else {
                            d0 = d0.min(d);
                        }
                    }
                    llr[nslot * level + 2 * n + ax] = ((d1 - d0) / (2.0 * var)).clamp(-1000.0, 1000.0);
                }
            }
        }
        llr.truncate(n_bits);
        hard.truncate(n_bits);
        (llr, hard, var)
    }
}
