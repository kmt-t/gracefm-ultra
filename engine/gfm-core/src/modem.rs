//! 変調（送信）と復調（受信）。GraceFM `gracefm/modem.py` に相当。QPSK と階層 16QAM。Y 方式は未対応。浮動小数点（信号点の平均エネルギーは 1）。

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
