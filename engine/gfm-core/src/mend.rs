//! MendNet の学習データ生成: 音源 → 符号化 → 通信路 → 実際の Decoder（ヘッダの事前知識とリスト復号つき）→
//! フレームごとの状態、復号した係数、誤差分散、ヘッダ採用の信頼度。人工の欠落パターンではなく、実際の復号の結果を使う。
//!
//! 出力（`MendDataset::write` の .npz）は、シナリオ s、フレーム f、係数 nc（= 成分数 × 64。成分ごとに帯域 k = 0〜4 の係数を並べた順）、
//! 成分数 ncomp、帯域数 nb について:
//! - `truth` (f, nc) f32: 原音の Haar 係数（満点 = 1.0）
//! - `dec` (s, f, nc) f32: Decoder が復号した係数（融合・補間の前。使えないフレームは 0）
//! - `status` (s, f) u8: 0 = 正常、1 = 高域欠落（基本 CRC は通った）、2 = 壊れた（届いたが基本 CRC が通らない）、3 = 欠落
//! - `var` (s, f, ncomp, nb) f32: 復号した係数の誤差分散（線形、満点 1.0 に対する二乗）。情報なしは -1
//! - `hint_h` (s, f, nc) f32: 壊れたフレーム（status 2）の尺度なしの手がかり（仮数の期待値 / qmax。`decoder::Hint`）。それ以外は 0
//! - `hint_q` (s, f, ncomp, nb) f32: 手がかりの確からしさ（仮数の事後標準偏差 / qmax）。それ以外は 0
//! - `gap` (s, f) f32: ヘッダ採用の、最尤との対数尤度の差 [nat]
//! - `llr` (s, f, nbit) i8: 受信 LLR（`gen` と同じ定義。欠落フレームは 0）、`bits` (f, nbit) u8: 送信した仮数ビット、
//!   `step` (f, nc) f32: 係数ごとの量子化の刻み（満点 = 2^22 の尺度）。ビットの属性（`pos` `coef_of_bit` `strong` `protected` `group`）は `gen` と同じ
//! - `esn0` (s, f) f32、`scen` (s, 4) f32、`start` i64

use crate::channel::{awgn_adc, Link, Scen};
use crate::codec::analyze;
use crate::decoder::{Decoder, Status};
use crate::gen::{parallel_map, quantized_llr, transmit, BitAttrs, Transmitted};
use crate::layout::Layout;
use crate::modem::Modem;
use crate::npz::{write_npz, Npy};
use crate::prior::PriorConfig;
use crate::profiles::Profile;
use crate::scrambler::Scrambler;

struct Scenario {
    dec: Vec<f32>,
    status: Vec<u8>,
    var: Vec<f32>,
    gap: Vec<f32>,
    llr: Vec<i8>,
    hint_h: Vec<f32>,
    hint_q: Vec<f32>,
    esn0: Vec<f32>,
}

pub struct MendDataset {
    nfr: usize,
    ncoef: usize,
    ncomp: usize,
    nbands: usize,
    scens: Vec<Scen>,
    truth: Vec<f32>,
    tx: Transmitted,
    attrs: BitAttrs,
    rx: Vec<Scenario>,
}

/// 成分ごとに帯域 k = 0〜4 の係数を並べた (ncomp × 64) 個。送らない成分（S のないプロファイル）は含めない
fn flatten(bands: &[Vec<Vec<f64>>], ncomp: usize) -> Vec<f32> {
    bands.iter().take(ncomp).flat_map(|ch| ch.iter().flatten().map(|&v| v as f32)).collect()
}

fn status_code(s: Status) -> u8 {
    match s {
        Status::Ok => 0,
        Status::HighFail => 1,
        Status::CrcFail => 2,
        Status::Lost => 3,
    }
}

/// pcm（満点 ±1 の L/R）から、各シナリオのデータを作る
pub fn generate_mend(p: &Profile, pcm: &[[f64; 2]], scens: &[Scen], seed: u64, threads: usize) -> Result<MendDataset, String> {
    let b = p.block;
    let nfr = pcm.len() / b;
    let ncomp = if p.bits_s.iter().any(|&v| v > 0) { 2 } else { 1 };
    let nb = p.nbands();
    let lay = Layout::new(p)?;
    let modem = Modem::new(p.qam16h, p.alpha);
    let scr = Scrambler::new(modem.n_symbols(lay.n_tx, lay.n_strong) * modem.bits_per_symbol());
    let tx = transmit(p, &lay, &pcm[..nfr * b], false)?;
    let nbit = lay.n_coef_bits();
    let decoder = Decoder::new(p)?;
    let truth: Vec<f32> = (0..nfr).flat_map(|i| flatten(&analyze(&pcm[i * b..(i + 1) * b], p), ncomp)).collect();
    let ncoef = truth.len() / nfr.max(1);
    let rx = parallel_map(scens.len(), threads, |si| {
        let mut link = Link::new(&scens[si], nfr, seed, si as u64);
        let received: Vec<Option<Vec<[f64; 2]>>> =
            tx.syms.iter().enumerate().map(|(i, f)| (!link.lost[i]).then(|| awgn_adc(f, link.esn0[i], &mut link.noise))).collect();
        let r = decoder.run(&received, nfr * b, 0, &PriorConfig::FULL);
        let mut s = Scenario {
            dec: Vec::with_capacity(nfr * ncoef),
            status: r.status.iter().map(|&s| status_code(s)).collect(),
            var: Vec::with_capacity(nfr * ncomp * nb),
            gap: r.gap.iter().map(|&g| g as f32).collect(),
            llr: vec![0; nfr * nbit],
            hint_h: Vec::with_capacity(nfr * ncoef),
            hint_q: Vec::with_capacity(nfr * ncomp * nb),
            esn0: link.esn0.iter().map(|&e| e as f32).collect(),
        };
        for (i, rx) in received.iter().enumerate() {
            if let Some(rx) = rx {
                let (q, _) = quantized_llr(&lay, &modem, &scr, rx);
                for (j, &idx) in lay.coef_stream_idx.iter().enumerate() {
                    s.llr[i * nbit + j] = q[idx];
                }
            }
            let hint = r.hint[i].as_ref();
            for ch in 0..ncomp {
                s.hint_h.extend(hint.map_or(vec![0.0; 64], |h| h.h[ch].iter().map(|&v| v as f32).collect()));
                s.hint_q.extend((0..nb).map(|k| hint.map_or(0.0, |h| h.q[ch][k] as f32)));
            }
            s.dec.extend(flatten(&analyze(&r.pcm[i * b..(i + 1) * b], p), ncomp));
            // 半ブロックごとの分散を、フレームで平均する。どちらかが情報なしなら、情報なし（-1）
            let (h0, h1) = (&r.rb[i * b], &r.rb[i * b + b / 2]);
            for ch in 0..ncomp {
                for k in 0..nb {
                    let v = (h0[ch][k] + h1[ch][k]) / 2.0;
                    s.var.push(if v.is_finite() { v as f32 } else { -1.0 });
                }
            }
        }
        s
    });
    Ok(MendDataset { nfr, ncoef, ncomp, nbands: nb, scens: scens.to_vec(), truth, tx, attrs: BitAttrs::new(p, &lay), rx })
}

impl MendDataset {
    /// .npz に書く。start: 音源の中の開始位置（記録用）
    pub fn write(&self, path: &str, start: i64) -> std::io::Result<()> {
        let (s, f, nc) = (self.rx.len(), self.nfr, self.ncoef);
        let cat = |get: &dyn Fn(&Scenario) -> &Vec<f32>| -> Vec<f32> { self.rx.iter().flat_map(|r| get(r).iter().copied()).collect() };
        let status: Vec<u8> = self.rx.iter().flat_map(|r| r.status.clone()).collect();
        let scen: Vec<f32> = self.scens.iter().flat_map(|c| [c.e0 as f32, c.fade as u8 as f32, c.rate as f32, c.burst as f32]).collect();
        let llr: Vec<i8> = self.rx.iter().flat_map(|r| r.llr.iter().copied()).collect();
        let nbit = self.tx.bits.len() / f;
        let mut entries = vec![
            ("truth", Npy::f32(&self.truth, &[f, nc])),
            ("dec", Npy::f32(&cat(&|r| &r.dec), &[s, f, nc])),
            ("status", Npy::u8(&status, &[s, f])),
            ("var", Npy::f32(&cat(&|r| &r.var), &[s, f, self.ncomp, self.nbands])),
            ("hint_h", Npy::f32(&cat(&|r| &r.hint_h), &[s, f, nc])),
            ("hint_q", Npy::f32(&cat(&|r| &r.hint_q), &[s, f, self.ncomp, self.nbands])),
            ("gap", Npy::f32(&cat(&|r| &r.gap), &[s, f])),
            ("llr", Npy::i8(&llr, &[s, f, nbit])),
            ("bits", Npy::u8(&self.tx.bits, &[f, nbit])),
            ("step", Npy::f32(&self.tx.step, &[f, nc])),
            ("esn0", Npy::f32(&cat(&|r| &r.esn0), &[s, f])),
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
            for i in 0..nfr {
                let l = &s.llr[i * nbit..(i + 1) * nbit];
                if s.status[i] == 3 {
                    lost += 1;
                    assert!(l.iter().all(|&v| v == 0));
                    continue;
                }
                wrong += l.iter().zip(&ds.tx.bits[i * nbit..]).filter(|(&v, &b)| v != 0 && ((v < 0) != (b == 1))).count();
            }
            assert!(wrong < nfr * nbit / 1000, "シナリオ {r}: 符号の不一致 {wrong}");
            assert_eq!(lost > 0, r == 1, "シナリオ {r}: 欠落 {lost}");
        }
    }
}
