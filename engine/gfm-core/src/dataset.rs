//! MendNet の学習データ生成: 音源 → 符号化 → 通信路 → 実際の Decoder（ヘッダの事前知識とリスト復号つき）→
//! MendNet の観測（`restore::Observation`）。人工の欠落パターンではなく、実際の復号の結果を使う。
//!
//! 出力（`MendDataset::write` の .npz）は、シナリオ s、フレーム f、係数 nc（= 成分数 × 64。成分ごとに帯域 k = 0〜4 の係数を並べた順）、
//! 成分数 ncomp、帯域数 nb について:
//! - `truth` (f, nc) f32: 原音の Haar 係数（満点 = 1.0）
//! - `dec` (s, f, nc) f32: Decoder が復号した係数（補間の前。使えないフレームは 0）
//! - `status` (s, f) u8: 0 = 正常、1 = 高域欠落（基本 CRC は通った）、2 = 壊れた（届いたが基本 CRC が通らない）、3 = 欠落
//! - `var` (s, f, ncomp, nb) f32: 復号した係数の誤差分散（線形、満点 1.0 に対する二乗）。情報なしは -1
//! - `hint_h` (s, f, nc) f32: 壊れたフレーム（status 2）の尺度なしの手がかり（仮数の期待値 / qmax。`decoder::Hint`）。それ以外は 0
//! - `hint_q` (s, f, ncomp, nb) f32: 手がかりの確からしさ（仮数の事後標準偏差 / qmax）。それ以外は 0
//! - `gap` (s, f) f32: ヘッダ採用の、最尤との対数尤度の差 [nat]
//! - `llr` (s, f, nbit) i8: 受信 LLR（自然対数の LLR の 4 倍、±127 に飽和。欠落フレームは 0）、`bits` (f, nbit) u8: 送信した仮数ビット、
//!   `step` (f, nc) f32: 係数ごとの量子化の刻み（満点 = 2^22 の尺度）、`dstep` (s, f, nc) f32: Decoder が復号したヘッダの刻み（基本 CRC が通らない・欠落のフレームと、高域欠落のフレームの高域は 0）。ビットの属性: `pos`（仮数内の位置）、`coef_of_bit`（係数の番号）、`strong`（強い位置か）、`protected`（FEC で守る上位ビットか）、`group`（係数の（成分, 帯域）の番号）
//! - `esn0` (s, f) f32、`scen` (s, 4) f32、`start` i64

use crate::channel::{awgn_adc, Link, Scen};
use crate::codec::analyze;
use crate::decoder::{block_steps, Decoder};
use crate::encoder::Encoder;
use crate::layout::Layout;
use crate::npz::{write_npz, Npy};
use crate::prior::PriorConfig;
use crate::profiles::Profile;
use crate::restore::Observation;

/// 送信側の結果（シナリオに依らない）
struct Transmitted {
    /// (F, nbit) 送信した仮数ビット（スクランブル前）
    bits: Vec<u8>,
    /// (F, nc) 係数ごとの量子化の刻み（満点 = 2^22 の尺度）
    step: Vec<f32>,
    syms: Vec<Vec<[f64; 2]>>,
}

fn transmit(p: &Profile, lay: &Layout, pcm: &[[f64; 2]]) -> Result<Transmitted, String> {
    let nfr = pcm.len() / p.block;
    let (nbit, ncoef) = (lay.n_coef_bits(), lay.n_coef);
    let mut enc = Encoder::new(p, 0, 0)?;
    let mut t = Transmitted { bits: vec![0; nfr * nbit], step: vec![0.0; nfr * ncoef], syms: Vec::with_capacity(nfr) };
    for i in 0..nfr {
        let (bd, stream, sym) = enc.frame_parts(&pcm[i * p.block..(i + 1) * p.block]);
        t.syms.push(sym);
        for (j, &idx) in lay.coef_stream_idx.iter().enumerate() {
            t.bits[i * nbit + j] = stream[idx];
        }
        block_steps(lay, &bd, true, &mut t.step[i * ncoef..(i + 1) * ncoef]);
    }
    Ok(t)
}

/// 仮数ビット・係数の属性（`pos` `coef_of_bit` `strong` `protected` `group`）
struct BitAttrs {
    pos: Vec<i64>,
    coef_of_bit: Vec<i64>,
    strong: Vec<u8>,
    protected: Vec<u8>,
    group: Vec<i64>,
}

impl BitAttrs {
    fn new(p: &Profile, lay: &Layout) -> BitAttrs {
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

    fn entries(&self) -> Vec<(&'static str, Npy)> {
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

struct Scenario {
    obs: Vec<Observation>,
    esn0: Vec<f32>,
}

pub struct MendDataset {
    nfr: usize,
    ncoef: usize,
    nbands: usize,
    scens: Vec<Scen>,
    truth: Vec<f32>,
    tx: Transmitted,
    attrs: BitAttrs,
    rx: Vec<Scenario>,
}

/// pcm（満点 ±1 の L/R）から、各シナリオのデータを作る。M・S の 2 成分のプロファイル用
pub fn generate_mend(p: &Profile, pcm: &[[f64; 2]], scens: &[Scen], seed: u64, threads: usize) -> Result<MendDataset, String> {
    let b = p.block;
    let nfr = pcm.len() / b;
    let lay = Layout::new(p)?;
    let tx = transmit(p, &lay, &pcm[..nfr * b])?;
    let truth: Vec<f32> =
        (0..nfr).flat_map(|i| analyze(&pcm[i * b..(i + 1) * b], p).into_iter().flatten().flatten().map(|v| v as f32)).collect();
    let rx = parallel_map(scens.len(), threads, |si| {
        let mut link = Link::new(&scens[si], nfr, seed, si as u64);
        let mut dec = Decoder::new(p, PriorConfig::FULL).expect("プロファイルは検査済み");
        let obs = (0..nfr)
            .map(|i| {
                let rx = (!link.lost[i]).then(|| awgn_adc(&tx.syms[i], link.esn0[i], &mut link.noise));
                Observation::new(&lay, &dec.push(rx.as_deref()))
            })
            .collect();
        Scenario { obs, esn0: link.esn0.iter().map(|&e| e as f32).collect() }
    });
    Ok(MendDataset { nfr, ncoef: lay.n_coef, nbands: p.nbands(), scens: scens.to_vec(), truth, tx, attrs: BitAttrs::new(p, &lay), rx })
}

/// f(0), f(1), ..., f(n-1) を threads 本のスレッドで計算して、順に返す
fn parallel_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
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

impl MendDataset {
    /// .npz に書く。start: 音源の中の開始位置（記録用）
    pub fn write(&self, path: &str, start: i64) -> std::io::Result<()> {
        let (s, f, nc) = (self.rx.len(), self.nfr, self.ncoef);
        let cat = |get: &dyn Fn(&Observation) -> &Vec<f32>| -> Vec<f32> {
            self.rx.iter().flat_map(|r| r.obs.iter().flat_map(|o| get(o).iter().copied())).collect()
        };
        let status: Vec<u8> = self.rx.iter().flat_map(|r| r.obs.iter().map(|o| o.status)).collect();
        let gap: Vec<f32> = self.rx.iter().flat_map(|r| r.obs.iter().map(|o| o.gap)).collect();
        let scen: Vec<f32> = self.scens.iter().flat_map(|c| [c.e0 as f32, c.fade as u8 as f32, c.rate as f32, c.burst as f32]).collect();
        let llr: Vec<i8> = self.rx.iter().flat_map(|r| r.obs.iter().flat_map(|o| o.llr.iter().copied())).collect();
        let nbit = self.tx.bits.len() / f;
        let mut entries = vec![
            ("truth", Npy::f32(&self.truth, &[f, nc])),
            ("dec", Npy::f32(&cat(&|r| &r.coef), &[s, f, nc])),
            ("status", Npy::u8(&status, &[s, f])),
            ("var", Npy::f32(&cat(&|r| &r.var), &[s, f, 2, self.nbands])),
            ("hint_h", Npy::f32(&cat(&|r| &r.hint_h), &[s, f, nc])),
            ("hint_q", Npy::f32(&cat(&|r| &r.hint_q), &[s, f, 2, self.nbands])),
            ("gap", Npy::f32(&gap, &[s, f])),
            ("llr", Npy::i8(&llr, &[s, f, nbit])),
            ("bits", Npy::u8(&self.tx.bits, &[f, nbit])),
            ("step", Npy::f32(&self.tx.step, &[f, nc])),
            ("dstep", Npy::f32(&cat(&|r| &r.dstep), &[s, f, nc])),
            ("esn0", Npy::f32(&self.rx.iter().flat_map(|r| r.esn0.iter().copied()).collect::<Vec<_>>(), &[s, f])),
            ("scen", Npy::f32(&scen, &[s, 4])),
            ("start", Npy::i64(&[start], &[])),
        ];
        entries.extend(self.attrs.entries());
        write_npz(path, &entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::profile_by_name;

    /// 高 Es/N0 では、受信 LLR の符号が送信した仮数ビットと一致する。欠落フレームの LLR は 0
    #[test]
    fn llr_agrees_with_transmitted_bits() {
        let p = profile_by_name("B200").unwrap();
        let nfr = 200;
        let pcm: Vec<[f64; 2]> = (0..nfr * p.block)
            .map(|n| [0.4 * (n as f64 * 0.05).sin() + 0.1 * (n as f64 * 0.9).sin(), 0.3 * (n as f64 * 0.03).cos()])
            .collect();
        let scens = [Scen { e0: 30.0, fade: false, rate: 0.0, burst: 1.0 }, Scen { e0: 30.0, fade: false, rate: 0.2, burst: 3.0 }];
        let ds = generate_mend(&p, &pcm, &scens, 1, 1).unwrap();
        let nbit = ds.tx.bits.len() / nfr;
        for (r, s) in ds.rx.iter().enumerate() {
            let (mut wrong, mut lost) = (0, 0);
            for (i, o) in s.obs.iter().enumerate() {
                let l = &o.llr;
                if o.status == 3 {
                    lost += 1;
                    assert!(l.iter().all(|&v| v == 0));
                    continue;
                }
                let nc = ds.ncoef;
                let (d, t) = (&o.dstep, &ds.tx.step[i * nc..(i + 1) * nc]);
                if o.status == 0 {
                    assert_eq!(d, t, "復号したヘッダの刻みが送信と違う");
                } else if o.status == 1 {
                    assert!(d.iter().zip(t).all(|(&d, &t)| d == 0.0 || d == t), "高域欠落のフレームの刻みが送信と違う");
                }
                wrong += l.iter().zip(&ds.tx.bits[i * nbit..]).filter(|(&v, &b)| v != 0 && ((v < 0) != (b == 1))).count();
            }
            assert!(wrong < nfr * nbit / 1000, "シナリオ {r}: 符号の不一致 {wrong}");
            assert_eq!(lost > 0, r == 1, "シナリオ {r}: 欠落 {lost}");
        }
    }
}
