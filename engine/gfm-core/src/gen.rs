//! HaarNet の学習データ生成: 音源 → 符号化 → 通信路シナリオ → 受信 LLR。
//!
//! 出力（`Dataset::write` の .npz）は、シナリオ s・フレーム f・仮数ビット nb・係数 nc について:
//! - `llr` (s, f, nb) i8: 受信 LLR（自然対数の LLR の 4 倍、±127 に飽和。欠落フレームは 0）
//! - `bits` (f, nb) u8: 送信した仮数ビット（スクランブル前）
//! - `step` (f, nc) f32: 係数ごとの量子化の刻み（満点 = 2^22 の尺度）
//! - `coef0` / `coefq` (f, nc) f32: 量子化前 / 後の係数（同じ尺度）。係数の並びは Layout の coef_offset 順
//! - `lost` (s, f) bool、`esn0` (s, f) f32（フェージング込み）、`var` (s, f) i64（軸あたりの雑音分散。(Q22 の距離 / 64)² の単位）、`scen` (s, 4) f32
//! - ビットの属性: `pos`（仮数内の位置）、`coef_of_bit`（係数の番号）、`strong`（強い位置か）、`protected`（FEC で守る上位ビットか）、`group`（係数の（成分, 帯域）の番号）

use crate::channel::{awgn_adc, Link, Scen};
use crate::codec::quant::BlockData;
use crate::codec::{analyze, exp_range};
use crate::decoder::reconstruct_coeffs;
use crate::encoder::Encoder;
use crate::layout::Layout;
use crate::modem::{Modem, N_PREAMBLE};
use crate::npz::{write_npz, Npy};
use crate::profiles::Profile;
use crate::scrambler::Scrambler;

const LLR_SCALE: f64 = 4.0;
const VAR_UNIT: f64 = (1u64 << 32) as f64;
const Q22: f64 = (1u64 << 22) as f64;

/// 送信側の結果（シナリオに依らない）
pub(crate) struct Transmitted {
    pub bits: Vec<u8>,  // (F, nbit)
    bits_full: Vec<u8>, // (F, n_tx)。full のときだけ
    pub step: Vec<f32>,
    coef0: Vec<f32>,
    coefq: Vec<f32>,
    pub syms: Vec<Vec<[f64; 2]>>,
}

/// 1 つの通信路シナリオの受信結果
struct Received {
    llr: Vec<i8>,      // (F, nbit)
    llr_full: Vec<i8>, // (F, n_tx)。full のときだけ
    lost: Vec<bool>,
    esn0: Vec<f32>,
    var: Vec<i64>,
}

/// 仮数ビット・係数の属性（`pos` `coef_of_bit` `strong` `protected` `group`）
pub(crate) struct BitAttrs {
    pos: Vec<i64>,
    coef_of_bit: Vec<i64>,
    strong: Vec<u8>,
    protected: Vec<u8>,
    group: Vec<i64>,
}

impl BitAttrs {
    pub(crate) fn new(p: &Profile, lay: &Layout) -> BitAttrs {
        let mut protected = vec![0u8; lay.n_coef_bits()];
        for &i in &lay.top_coef_idx {
            protected[i] = 1;
        }
        let mut group = vec![0i64; lay.n_coef];
        for (gi, &(ch, k)) in lay.coef_groups.iter().enumerate() {
            let o = lay.coef_offset[&(ch, k)];
            group[o..o + p.ncoef(k)].fill(gi as i64);
        }
        BitAttrs {
            pos: lay.coef_pos.clone(),
            coef_of_bit: lay.coef_flat.iter().map(|&v| v as i64).collect(),
            strong: lay.coef_stream_idx.iter().map(|&i| (i < lay.n_strong) as u8).collect(),
            protected,
            group,
        }
    }

    pub(crate) fn entries(&self) -> Vec<(&'static str, Npy)> {
        let (nb, nc) = (self.pos.len(), self.group.len());
        vec![
            ("pos", Npy::i64(&self.pos, &[nb])),
            ("coef_of_bit", Npy::i64(&self.coef_of_bit, &[nb])),
            ("strong", Npy::u8(&self.strong, &[nb])),
            ("protected", Npy::u8(&self.protected, &[nb])),
            ("group", Npy::i64(&self.group, &[nc])),
        ]
    }
}

/// 生成した学習データ
pub struct Dataset {
    pub nfr: usize,
    pub nbit: usize,
    pub ncoef: usize,
    pub n_tx: usize,
    pub scens: Vec<Scen>,
    tx: Transmitted,
    rx: Vec<Received>,
    attrs: BitAttrs,
}

/// BlockData（指数とアテネーション）から、係数ごとの量子化の刻み（満点 = 2^22 の尺度）を out に書く。指数やアテネーションがない帯域は 0 のまま
pub(crate) fn block_steps(lay: &Layout, bd: &BlockData, out: &mut [f32]) {
    let p = &lay.p;
    for &(ch, k) in &lay.coef_groups {
        let (Some(e), Some(attn)) = (bd.eidx[ch][k], bd.attn[ch][k].as_ref()) else { continue };
        let e = e + exp_range(k, p.bits(ch)[k], p).0;
        let o = lay.coef_offset[&(ch, k)];
        for j in 0..p.ncoef(k) {
            out[o + j] = (2f64.powi((e - p.attn_at(ch, k, attn, j)) as i32) * Q22) as f32;
        }
    }
}

pub(crate) fn transmit(p: &Profile, lay: &Layout, pcm: &[[f64; 2]], full: bool) -> Result<Transmitted, String> {
    let nfr = pcm.len() / p.block;
    let (nbit, ncoef) = (lay.n_coef_bits(), lay.n_coef);
    let mut enc = Encoder::new(p, 0, 0)?;
    let mut t = Transmitted {
        bits: vec![0; nfr * nbit],
        bits_full: if full { vec![0; nfr * lay.n_tx] } else { Vec::new() },
        step: vec![0.0; nfr * ncoef],
        coef0: vec![0.0; nfr * ncoef],
        coefq: vec![0.0; nfr * ncoef],
        syms: Vec::with_capacity(nfr),
    };
    for i in 0..nfr {
        let blk = &pcm[i * p.block..(i + 1) * p.block];
        let (bd, stream, sym) = enc.frame_parts(blk);
        t.syms.push(sym);
        for (j, &idx) in lay.coef_stream_idx.iter().enumerate() {
            t.bits[i * nbit + j] = stream[idx];
        }
        if full {
            t.bits_full[i * lay.n_tx..(i + 1) * lay.n_tx].copy_from_slice(&stream);
        }
        let bands = analyze(blk, p);
        let quantized = reconstruct_coeffs(lay, &bd);
        block_steps(lay, &bd, &mut t.step[i * ncoef..(i + 1) * ncoef]);
        for &(ch, k) in &lay.coef_groups {
            let o = i * ncoef + lay.coef_offset[&(ch, k)];
            for j in 0..p.ncoef(k) {
                t.coef0[o + j] = (bands[ch][k][j] * Q22) as f32;
                t.coefq[o + j] = (quantized[ch][k][j] * Q22) as f32;
            }
        }
    }
    Ok(t)
}

/// 受信シンボル（プリアンブルつき）を復調して、スクランブルを戻した LLR（自然対数の 4 倍、±127 に飽和）を送信位置の順に返す。軸あたりの雑音分散も返す
pub(crate) fn quantized_llr(lay: &Layout, modem: &Modem, scr: &Scrambler, rx: &[[f64; 2]]) -> (Vec<i8>, f64) {
    let (llr, _, var) = modem.demodulate(&rx[N_PREAMBLE..], lay.n_tx);
    let q = (0..lay.n_tx).map(|i| ((if scr.sign(i) { -llr[i] } else { llr[i] }) * LLR_SCALE).round().clamp(-127.0, 127.0) as i8).collect();
    (q, var)
}

/// 受信側の共通の道具（全シナリオで共有する）
struct Receiver<'a> {
    lay: &'a Layout,
    modem: &'a Modem,
    scr: &'a Scrambler,
    tx: &'a Transmitted,
    seed: u64,
    full: bool,
}

impl Receiver<'_> {
    /// シナリオ si の通信路を通した受信 LLR
    fn scenario(&self, sc: &Scen, si: usize) -> Received {
        let (lay, nfr, nbit) = (self.lay, self.tx.syms.len(), self.lay.n_coef_bits());
        let mut link = Link::new(sc, nfr, self.seed, si as u64);
        let mut r = Received {
            llr: vec![0; nfr * nbit],
            llr_full: if self.full { vec![0; nfr * lay.n_tx] } else { Vec::new() },
            lost: link.lost.clone(),
            esn0: link.esn0.iter().map(|&e| e as f32).collect(),
            var: vec![0; nfr],
        };
        for (i, sym) in self.tx.syms.iter().enumerate() {
            if link.lost[i] {
                continue;
            }
            let rx = awgn_adc(sym, link.esn0[i], &mut link.noise); // プリアンブルにも雑音を加える（gfm-tx と同じ実現になる）
            let (q, var) = quantized_llr(lay, self.modem, self.scr, &rx);
            r.var[i] = ((var * VAR_UNIT).round() as i64).max(1);
            for (j, &idx) in lay.coef_stream_idx.iter().enumerate() {
                r.llr[i * nbit + j] = q[idx];
            }
            if self.full {
                r.llr_full[i * lay.n_tx..(i + 1) * lay.n_tx].copy_from_slice(&q);
            }
        }
        r
    }
}

/// f(0), f(1), ..., f(n-1) を threads 本のスレッドで計算して、順に返す
pub(crate) fn parallel_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = threads.clamp(1, n.max(1));
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        let f = &f;
        let handles: Vec<_> =
            (0..threads).map(|t| scope.spawn(move || (t..n).step_by(threads).map(|i| (i, f(i))).collect::<Vec<_>>())).collect();
        for h in handles {
            for (i, v) in h.join().expect("スレッドが異常終了しました") {
                out[i] = Some(v);
            }
        }
    });
    out.into_iter().map(|v| v.expect("全ての要素を計算したはず")).collect()
}

/// pcm（満点 ±1 の L/R）から、各シナリオの学習データを作る。full: 全送信位置の bits_full / llr_full も残す（試験用）
pub fn generate(p: &Profile, pcm: &[[f64; 2]], scens: &[Scen], seed: u64, threads: usize, full: bool) -> Result<Dataset, String> {
    let lay = Layout::new(p)?;
    let modem = Modem::new(p.qam16h, p.alpha);
    let scr = Scrambler::new(modem.n_symbols(lay.n_tx, lay.n_strong) * modem.bits_per_symbol());
    let tx = transmit(p, &lay, pcm, full)?;
    let receiver = Receiver { lay: &lay, modem: &modem, scr: &scr, tx: &tx, seed, full };
    let rx = parallel_map(scens.len(), threads, |si| receiver.scenario(&scens[si], si));
    Ok(Dataset {
        nfr: tx.syms.len(),
        nbit: lay.n_coef_bits(),
        ncoef: lay.n_coef,
        n_tx: lay.n_tx,
        scens: scens.to_vec(),
        attrs: BitAttrs::new(p, &lay),
        tx,
        rx,
    })
}

impl Dataset {
    /// .npz に書く。start: 音源の中の開始位置（記録用）
    pub fn write(&self, path: &str, start: i64) -> std::io::Result<()> {
        let (s, f, nb, nc) = (self.scens.len(), self.nfr, self.nbit, self.ncoef);
        let cat = |get: &dyn Fn(&Received) -> Vec<i8>| -> Vec<i8> { self.rx.iter().flat_map(get).collect() };
        let llr = cat(&|r| r.llr.clone());
        let lost: Vec<bool> = self.rx.iter().flat_map(|r| r.lost.clone()).collect();
        let esn0: Vec<f32> = self.rx.iter().flat_map(|r| r.esn0.clone()).collect();
        let var: Vec<i64> = self.rx.iter().flat_map(|r| r.var.clone()).collect();
        let scen: Vec<f32> = self.scens.iter().flat_map(|c| [c.e0 as f32, c.fade as u8 as f32, c.rate as f32, c.burst as f32]).collect();
        let mut entries: Vec<(&str, Npy)> = vec![
            ("llr", Npy::i8(&llr, &[s, f, nb])),
            ("bits", Npy::u8(&self.tx.bits, &[f, nb])),
            ("step", Npy::f32(&self.tx.step, &[f, nc])),
            ("coef0", Npy::f32(&self.tx.coef0, &[f, nc])),
            ("coefq", Npy::f32(&self.tx.coefq, &[f, nc])),
            ("lost", Npy::bool(&lost, &[s, f])),
            ("esn0", Npy::f32(&esn0, &[s, f])),
            ("var", Npy::i64(&var, &[s, f])),
            ("scen", Npy::f32(&scen, &[s, 4])),
            ("start", Npy::i64(&[start], &[])),
        ];
        entries.extend(self.attrs.entries());
        if !self.tx.bits_full.is_empty() {
            let llr_full = cat(&|r| r.llr_full.clone());
            entries.push(("bits_full", Npy::u8(&self.tx.bits_full, &[f, self.n_tx])));
            entries.push(("llr_full", Npy::i8(&llr_full, &[s, f, self.n_tx])));
        }
        write_npz(path, &entries)
    }
}
