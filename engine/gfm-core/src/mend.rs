//! MendNet の学習データ生成: 音源 → 符号化 → 通信路 → 実際の Decoder（ヘッダの事前知識とリスト復号つき）→
//! フレームごとの状態、復号した係数、誤差分散、ヘッダ採用の信頼度。人工の欠落パターンではなく、実際の復号の結果を使う。
//!
//! 出力（`MendDataset::write` の .npz）は、シナリオ s、フレーム f、係数 nc（= 成分数 × 64。成分ごとに帯域 k = 0〜4 の係数を並べた順）、
//! 成分数 ncomp、帯域数 nb について:
//! - `truth` (f, nc) f32: 原音の Haar 係数（満点 = 1.0）
//! - `dec` (s, f, nc) f32: Decoder が復号した係数（融合・補間の前。使えないフレームは 0）
//! - `status` (s, f) u8: 0 = 正常、1 = 高域欠落（基本 CRC は通った）、2 = 壊れた（届いたが基本 CRC が通らない）、3 = 欠落
//! - `var` (s, f, ncomp, nb) f32: 復号した係数の誤差分散（線形、満点 1.0 に対する二乗）。情報なしは -1
//! - `gap` (s, f) f32: ヘッダ採用の、最尤との対数尤度の差 [nat]
//! - `esn0` (s, f) f32、`scen` (s, 4) f32、`start` i64

use crate::channel::{awgn_adc, Link, Scen};
use crate::codec::analyze;
use crate::decoder::{Decoder, Status};
use crate::encoder::Encoder;
use crate::gen::parallel_map;
use crate::npz::{write_npz, Npy};
use crate::prior::PriorConfig;
use crate::profiles::Profile;

struct Scenario {
    dec: Vec<f32>,
    status: Vec<u8>,
    var: Vec<f32>,
    gap: Vec<f32>,
    esn0: Vec<f32>,
}

pub struct MendDataset {
    nfr: usize,
    ncoef: usize,
    ncomp: usize,
    nbands: usize,
    scens: Vec<Scen>,
    truth: Vec<f32>,
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
    let mut enc = Encoder::new(p, 0, 0)?;
    let frames = enc.run(&pcm[..nfr * b]);
    let decoder = Decoder::new(p)?;
    let truth: Vec<f32> = (0..nfr).flat_map(|i| flatten(&analyze(&pcm[i * b..(i + 1) * b], p), ncomp)).collect();
    let ncoef = truth.len() / nfr.max(1);
    let rx = parallel_map(scens.len(), threads, |si| {
        let mut link = Link::new(&scens[si], nfr, seed, si as u64);
        let received: Vec<Option<Vec<[f64; 2]>>> =
            frames.iter().enumerate().map(|(i, f)| (!link.lost[i]).then(|| awgn_adc(f, link.esn0[i], &mut link.noise))).collect();
        let r = decoder.run(&received, nfr * b, 0, &PriorConfig::FULL);
        let mut s = Scenario {
            dec: Vec::with_capacity(nfr * ncoef),
            status: r.status.iter().map(|&s| status_code(s)).collect(),
            var: Vec::with_capacity(nfr * ncomp * nb),
            gap: r.gap.iter().map(|&g| g as f32).collect(),
            esn0: link.esn0.iter().map(|&e| e as f32).collect(),
        };
        for i in 0..nfr {
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
    Ok(MendDataset { nfr, ncoef, ncomp, nbands: nb, scens: scens.to_vec(), truth, rx })
}

impl MendDataset {
    /// .npz に書く。start: 音源の中の開始位置（記録用）
    pub fn write(&self, path: &str, start: i64) -> std::io::Result<()> {
        let (s, f, nc) = (self.rx.len(), self.nfr, self.ncoef);
        let cat = |get: &dyn Fn(&Scenario) -> &Vec<f32>| -> Vec<f32> { self.rx.iter().flat_map(|r| get(r).iter().copied()).collect() };
        let status: Vec<u8> = self.rx.iter().flat_map(|r| r.status.clone()).collect();
        let scen: Vec<f32> = self.scens.iter().flat_map(|c| [c.e0 as f32, c.fade as u8 as f32, c.rate as f32, c.burst as f32]).collect();
        write_npz(
            path,
            &[
                ("truth", Npy::f32(&self.truth, &[f, nc])),
                ("dec", Npy::f32(&cat(&|r| &r.dec), &[s, f, nc])),
                ("status", Npy::u8(&status, &[s, f])),
                ("var", Npy::f32(&cat(&|r| &r.var), &[s, f, self.ncomp, self.nbands])),
                ("gap", Npy::f32(&cat(&|r| &r.gap), &[s, f])),
                ("esn0", Npy::f32(&cat(&|r| &r.esn0), &[s, f])),
                ("scen", Npy::f32(&scen, &[s, 4])),
                ("start", Npy::i64(&[start], &[])),
            ],
        )
    }
}
