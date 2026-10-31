//! 学習データの生成: 音源 → 符号化 → 通信路シナリオ → 受信 LLR。tools/gen_dataset.py + tools/prep_haarnet.py と同じ内容を出す。

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::codec::{decode_coeffs, encode_block, exp_range, haar_forward, FRAC};
use crate::framing::pack;
use crate::layout::Layout;
use crate::modem::{prbs, Modem, N_PREAMBLE, SYM_FRAC};
use crate::npz::{write_npz, Npy};
use crate::profiles::Profile;

#[derive(Clone, Copy, Debug)]
pub struct Scen {
    pub e0: f64,
    pub fade: bool,
    pub rate: f64,
    pub burst: f64,
}

struct Gauss {
    rng: Pcg64,
    spare: Option<f64>,
}

impl Gauss {
    fn new(seed: u64) -> Gauss {
        Gauss { rng: Pcg64::seed_from_u64(seed), spare: None }
    }
    fn normal(&mut self) -> f64 {
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

/// フレームごとの電力利得（平均 1）。rayleigh: 複素利得を AR(1)（時定数 50 フレーム）で相関させる
fn fading_gains(nfr: usize, rayleigh: bool, g: &mut Gauss) -> Vec<f64> {
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

/// Gilbert-Elliott 型の欠落。rate: 欠落率、burst: 平均バースト長
fn loss_mask(nfr: usize, rate: f64, burst: f64, rng: &mut Pcg64) -> Vec<bool> {
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

const ADC_BITS: i64 = 12;
const ADC_ONE: f64 = (1 << (ADC_BITS - 2)) as f64;

/// 実数のシンボルに Es/N0 = snr_db の雑音を加え、12 bit の ADC で量子化して、小数 22 bit の整数に直す
fn awgn_adc(sym: &[[i64; 2]], snr_db: f64, g: &mut Gauss) -> Vec<[i64; 2]> {
    let n0 = 10f64.powf(-snr_db / 10.0);
    let std = (n0 / 2.0).sqrt();
    let one = (1u64 << SYM_FRAC) as f64;
    sym.iter()
        .map(|s| {
            let mut o = [0i64; 2];
            for ax in 0..2 {
                let v = s[ax] as f64 / one + std * g.normal();
                let code = (v * ADC_ONE).round_ties_even().clamp(-(1 << (ADC_BITS - 1)) as f64, ((1 << (ADC_BITS - 1)) - 1) as f64) as i64;
                o[ax] = code << (SYM_FRAC - ADC_BITS + 2);
            }
            o
        })
        .collect()
}

pub struct Output {
    pub profile: Profile,
    pub nfr: usize,
    pub nbit: usize,
    pub ncoef: usize,
    pub scens: Vec<Scen>,
    pub llr: Vec<i8>,       // (S, F, nbit)
    pub bits: Vec<u8>,      // (F, nbit)
    pub step: Vec<f32>,     // (F, ncoef)
    pub coef0: Vec<f32>,
    pub coefq: Vec<f32>,
    pub lost: Vec<bool>,    // (S, F)
    pub esn0: Vec<f32>,
    pub var: Vec<i64>,
    pub pos: Vec<i64>,
    pub coef_of_bit: Vec<i64>,
    pub strong: Vec<u8>,
    pub protected: Vec<u8>,
    pub group: Vec<i64>,
    pub bits_full: Vec<u8>, // (F, n_tx)
    pub llr_full: Vec<i8>,  // (S, F, n_tx)
    pub n_tx: usize,
}

pub fn generate(p: &Profile, pcm: &[[i64; 2]], scens: &[Scen], seed: u64, threads: usize, full: bool) -> Result<Output, String> {
    let lay = Layout::new(p)?;
    let modem = Modem::new(p.qam16h, p.alpha);
    let bps = modem.bits_per_symbol();
    let nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
    let nfr = pcm.len() / p.block;
    let (nbit, ncoef) = (lay.n_coef_bits(), lay.n_coef);
    let scr = prbs(nsym * bps);

    let mut bits = vec![0u8; nfr * nbit];
    let mut step = vec![0f32; nfr * ncoef];
    let mut coef0 = vec![0f32; nfr * ncoef];
    let mut coefq = vec![0f32; nfr * ncoef];
    let mut bits_full = if full { vec![0u8; nfr * lay.n_tx] } else { Vec::new() };
    let mut syms: Vec<Vec<[i64; 2]>> = Vec::with_capacity(nfr);
    for i in 0..nfr {
        let blk = &pcm[i * p.block..(i + 1) * p.block];
        let bd = encode_block(blk, p, true);
        let stream = pack(&lay, &bd, 0, i as u64);
        let mut buf = vec![0u8; nsym * bps];
        buf[..lay.n_tx].copy_from_slice(&stream);
        for (b, s) in buf.iter_mut().zip(&scr) {
            *b ^= *s;
        }
        syms.push(modem.modulate(&buf, nsym));
        for (j, &idx) in lay.coef_stream_idx.iter().enumerate() {
            bits[i * nbit + j] = stream[idx];
        }
        if full {
            bits_full[i * lay.n_tx..(i + 1) * lay.n_tx].copy_from_slice(&stream);
        }
        let ms = [
            blk.iter().map(|v| (v[0] + v[1] + 2) >> 2).collect::<Vec<_>>(),
            blk.iter().map(|v| (v[0] - v[1] + 2) >> 2).collect::<Vec<_>>(),
        ];
        let dq = decode_coeffs(&bd, p);
        for &(ch, k) in &lay.coef_groups {
            let o = lay.coef_offset[&(ch, k)];
            let b = p.bits(ch)[k];
            let bands = haar_forward(&ms[ch], p.levels);
            let (emin, _) = exp_range(k, b, p);
            let s = bd.eidx[ch][k].unwrap() + emin + FRAC;
            let attn = bd.attn[ch][k].as_ref().unwrap();
            let group = p.attenuator_group(ch, k);
            for j in 0..p.ncoef(k) {
                let a = if group > 0 { attn[j / group] } else { 0 };
                step[i * ncoef + o + j] = 2f64.powi((s - a) as i32) as f32;
                coef0[i * ncoef + o + j] = bands[k][j] as f32;
                coefq[i * ncoef + o + j] = dq[ch][k][j] as f32;
            }
        }
    }

    // 通信路: シナリオごと
    let ns = scens.len();
    let results: Vec<(Vec<i8>, Vec<i8>, Vec<bool>, Vec<f32>, Vec<i64>)> = {
        let work = |si: usize| -> (Vec<i8>, Vec<i8>, Vec<bool>, Vec<f32>, Vec<i64>) {
            let sc = scens[si];
            let mut gfade = Gauss::new(seed * 1000 + si as u64);
            let gains = fading_gains(nfr, sc.fade, &mut gfade);
            let mut lrng = Pcg64::seed_from_u64(seed * 1000 + 500 + si as u64);
            let lost = loss_mask(nfr, sc.rate, sc.burst, &mut lrng);
            let mut crng = Gauss::new(seed * 1000 + 900 + si as u64);
            let mut llr = vec![0i8; nfr * nbit];
            let mut llr_full = if full { vec![0i8; nfr * lay.n_tx] } else { Vec::new() };
            let mut esn0 = vec![0f32; nfr];
            let mut var = vec![0i64; nfr];
            for i in 0..nfr {
                let e = sc.e0 + 10.0 * gains[i].max(1e-6).log10();
                esn0[i] = e as f32;
                if lost[i] {
                    continue;
                }
                let rx = awgn_adc(&syms[i][N_PREAMBLE..], e, &mut crng);
                let (l, v) = modem.demodulate(&rx, lay.n_tx);
                var[i] = v;
                let fix = |q: usize| -> i8 { (if scr[q] == 1 { -l[q] } else { l[q] }).clamp(-127, 127) as i8 };
                for (j, &idx) in lay.coef_stream_idx.iter().enumerate() {
                    llr[i * nbit + j] = fix(idx);
                }
                if full {
                    for q in 0..lay.n_tx {
                        llr_full[i * lay.n_tx + q] = fix(q);
                    }
                }
            }
            (llr, llr_full, lost, esn0, var)
        };
        let threads = threads.max(1).min(ns.max(1));
        let mut res: Vec<Option<_>> = (0..ns).map(|_| None).collect();
        std::thread::scope(|sc| {
            let work = &work;
            let mut handles = Vec::new();
            for t in 0..threads {
                handles.push(sc.spawn(move || (t..ns).step_by(threads).map(|si| (si, work(si))).collect::<Vec<_>>()));
            }
            for h in handles {
                for (si, r) in h.join().unwrap() {
                    res[si] = Some(r);
                }
            }
        });
        res.into_iter().map(|r| r.unwrap()).collect()
    };

    let mut out = Output {
        profile: p.clone(), nfr, nbit, ncoef, scens: scens.to_vec(), llr: Vec::new(), bits, step, coef0, coefq,
        lost: Vec::new(), esn0: Vec::new(), var: Vec::new(), pos: lay.coef_pos.clone(),
        coef_of_bit: lay.coef_flat.iter().map(|&v| v as i64).collect(), strong: Vec::new(), protected: Vec::new(),
        group: vec![0; ncoef], bits_full, llr_full: Vec::new(), n_tx: lay.n_tx,
    };
    for (llr, llr_full, lost, esn0, var) in results {
        out.llr.extend(llr);
        out.llr_full.extend(llr_full);
        out.lost.extend(lost);
        out.esn0.extend(esn0);
        out.var.extend(var);
    }
    out.strong = lay.coef_stream_idx.iter().map(|&i| (i < lay.n_strong) as u8).collect();
    let mut prot = vec![0u8; nbit];
    for &i in &lay.top_coef_idx {
        prot[i] = 1;
    }
    out.protected = prot;
    for (gi, &(ch, k)) in lay.coef_groups.iter().enumerate() {
        let o = lay.coef_offset[&(ch, k)];
        for j in 0..p.ncoef(k) {
            out.group[o + j] = gi as i64;
        }
    }
    Ok(out)
}

impl Output {
    pub fn write(&self, path: &str, start: i64) -> std::io::Result<()> {
        let (s, f, nb, nc) = (self.scens.len(), self.nfr, self.nbit, self.ncoef);
        let scen: Vec<f32> = self.scens.iter().flat_map(|c| [c.e0 as f32, c.fade as u8 as f32, c.rate as f32, c.burst as f32]).collect();
        let mut entries: Vec<(&str, Npy)> = vec![
            ("llr", Npy::i8(&self.llr, &[s, f, nb])),
            ("bits", Npy::u8(&self.bits, &[f, nb])),
            ("step", Npy::f32(&self.step, &[f, nc])),
            ("coef0", Npy::f32(&self.coef0, &[f, nc])),
            ("coefq", Npy::f32(&self.coefq, &[f, nc])),
            ("lost", Npy::bool(&self.lost, &[s, f])),
            ("esn0", Npy::f32(&self.esn0, &[s, f])),
            ("var", Npy::i64(&self.var, &[s, f])),
            ("scen", Npy::f32(&scen, &[s, 4])),
            ("pos", Npy::i64(&self.pos, &[nb])),
            ("coef_of_bit", Npy::i64(&self.coef_of_bit, &[nb])),
            ("strong", Npy::u8(&self.strong, &[nb])),
            ("protected", Npy::u8(&self.protected, &[nb])),
            ("group", Npy::i64(&self.group, &[nc])),
            ("start", Npy::i64(&[start], &[])),
        ];
        if !self.bits_full.is_empty() {
            entries.push(("bits_full", Npy::u8(&self.bits_full, &[f, self.n_tx])));
            entries.push(("llr_full", Npy::i8(&self.llr_full, &[s, f, self.n_tx])));
        }
        write_npz(path, &entries)
    }
}
