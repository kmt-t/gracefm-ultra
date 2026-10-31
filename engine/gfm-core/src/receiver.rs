//! 出力段: 2 系統の融合、クロスフェード、欠落補間（PLC）、出力 IIR 切替。GraceFM `gracefm/txrx.py` の receive の移植。
//! フレームの復号（decoder.rs）とは分けた、送信に対応のない処理。

use crate::codec::quant::INF_LOG;
use crate::decoder::{Decoder, Stats};
use crate::fusion::fuse_band;
use crate::iir::apply_alt;
use crate::plc::{conceal, crossfade_switch, PlcParams};
use crate::profiles::Profile;

pub struct RxOutput {
    pub pcm: Vec<[i64; 2]>,
    pub quality: Vec<i64>,
    pub stats_a: Stats,
    pub stats_b: Option<Stats>,
    pub plc_samples: usize,
}

/// 出力のブロックごとの品質コード。2 系統のときは良い方の値を採る
pub fn block_quality(block: usize, qa: &[i64], qb: Option<&[i64]>, offset_b: usize, n_blocks: usize) -> Vec<i64> {
    (0..n_blocks)
        .map(|i| {
            let mut q = Vec::new();
            if i < qa.len() {
                q.push(qa[i]);
            }
            if let Some(qb) = qb {
                let j = (i * block + offset_b) / block;
                if j < qb.len() {
                    q.push(qb[j]);
                }
            }
            q.into_iter().min().unwrap_or(2)
        })
        .collect()
}

/// 2 系統（frames_b = None なら 1 系統）を受信して n サンプルの整数の音声を返す。副系は offset_b サンプル遅れて届く
pub fn receive(
    profile: &Profile, frames_a: &[Option<Vec<[i64; 2]>>], frames_b: Option<&[Option<Vec<[i64; 2]>>]>, n: usize, offset_b: usize,
    profile_b: Option<&Profile>,
) -> Result<RxOutput, String> {
    let pb = profile_b.unwrap_or(profile);
    if pb.fs != profile.fs || pb.block != profile.block {
        return Err("副系のサンプリング周波数とブロック長は主系と同じにしてください".into());
    }
    let mut da = Decoder::new(profile)?;
    let a = da.run(frames_a, n, 0);
    let mut db: Option<(Decoder, crate::decoder::StreamOutput)> = None;
    let (y, v) = if let Some(fb) = frames_b {
        let mut d = Decoder::new(pb)?;
        let b = d.run(fb, n, offset_b);
        let r = fuse_band(&a.pcm, &a.valid, &b.pcm, &b.valid, profile.block, profile.levels, &a.rb_trace, &b.rb_trace);
        db = Some((d, b));
        r
    } else {
        (a.pcm.clone(), a.valid.clone())
    };
    let plc_samples = v.iter().filter(|&&x| !x).count();
    let nblk = (y.len() + profile.block - 1) / profile.block;
    let quality = block_quality(profile.block, &a.quality, db.as_ref().map(|(_, b)| b.quality.as_slice()), offset_b, nblk);
    let mut traces = vec![&a.rb_trace];
    if let Some((_, b)) = &db {
        traces.push(&b.rb_trace);
    }
    let patterns: Vec<Vec<bool>> = (0..y.len() / profile.block)
        .map(|i| {
            let mut pat = Vec::new();
            for rb in &traces {
                let r = rb.get(i * profile.block).copied().unwrap_or([[INF_LOG; 5]; 2]);
                for ci in 0..2 {
                    for k in 0..profile.nbands() {
                        pat.push(r[ci][k] != INF_LOG);
                    }
                }
            }
            pat
        })
        .collect();
    let pp = PlcParams::default();
    let y = crossfade_switch(&y, &patterns, profile.block, &pp);
    let y = conceal(&y, &v, profile.fs as usize, &pp);
    let pcm = apply_alt(&y, &quality, profile.fs as usize, profile.block);
    Ok(RxOutput { pcm, quality, stats_a: da.stats, stats_b: db.map(|(d, _)| d.stats), plc_samples })
}
