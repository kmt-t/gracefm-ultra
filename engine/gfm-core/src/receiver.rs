//! 出力段: 2 系統の融合、クロスフェード、欠落補間（PLC）、出力 IIR 切替。GraceFM `gracefm/receiver.py` に相当。浮動小数点。
//! フレームの復号（decoder.rs）とは分けた、送信に対応のない処理。

use crate::decoder::{Decoder, Rb, Stats, QUALITY_MISSING};
use crate::fusion::fuse_band;
use crate::iir::apply_alt;
use crate::plc::{conceal, crossfade_switch, PlcParams};
use crate::prior::PriorConfig;
use crate::profiles::{Profile, MAX_BANDS};

pub struct RxOutput {
    pub pcm: Vec<[f64; 2]>,
    pub quality: Vec<i64>,
    pub stats_a: Stats,
    pub stats_b: Option<Stats>,
    pub plc_samples: usize,
}

/// 出力のブロックごとの品質コード（0 正常, 1 高域欠落, 2 欠落）。2 系統のときは良い方の系統の値を採る。
/// qa, qb: 各系統のフレームごとの品質コード（qb は 1 系統のとき None）。副系のフレームは、出力の時刻から offset_b を引いた位置で対応づける
pub fn block_quality(profile: &Profile, qa: &[i64], qb: Option<&[i64]>, offset_b: usize, n_blocks: usize) -> Vec<i64> {
    let b = profile.block;
    (0..n_blocks)
        .map(|i| {
            let mut q = Vec::new();
            if i < qa.len() {
                q.push(qa[i]);
            }
            if let Some(qb) = qb {
                let j = (i * b + offset_b) / b;
                if j < qb.len() {
                    q.push(qb[j]);
                }
            }
            q.into_iter().min().unwrap_or(QUALITY_MISSING)
        })
        .collect()
}

/// 合成後の音声 y と有効フラグ v を、欠落の補間と出力 IIR 切替にかける。
/// quality: 出力ブロックごとの品質コード、traces: 系統ごとの誤差分散（サンプルごと）
pub fn output_stage(profile: &Profile, y: &[[f64; 2]], v: &[bool], quality: &[i64], traces: &[&Vec<Rb>]) -> Vec<[f64; 2]> {
    // 情報のある（系統, 成分, 帯域）の組。これが変わったブロックの先頭を、クロスフェードでつなぐ
    let patterns: Vec<Vec<bool>> = (0..y.len() / profile.block)
        .map(|i| {
            traces
                .iter()
                .flat_map(|rb| rb.get(i * profile.block).copied().unwrap_or([[f64::INFINITY; MAX_BANDS]; 2]).into_iter().flatten())
                .map(f64::is_finite)
                .collect()
        })
        .collect();
    let pp = PlcParams::default();
    let y = crossfade_switch(y, &patterns, profile.block, &pp);
    let y = conceal(&y, v, profile.fs as usize, &pp);
    apply_alt(&y, quality, profile.fs as usize, profile.block)
}

/// 2 系統（frames_b = None なら 1 系統）を受信して n サンプルの音声（満点 ±1）を返す。副系は offset_b サンプル遅れて届く
pub fn receive(
    profile: &Profile,
    frames_a: &[Option<Vec<[f64; 2]>>],
    frames_b: Option<&[Option<Vec<[f64; 2]>>]>,
    n: usize,
    offset_b: usize,
    profile_b: Option<&Profile>,
    cfg: &PriorConfig,
) -> Result<RxOutput, String> {
    let pb = profile_b.unwrap_or(profile);
    if pb.fs != profile.fs || pb.block != profile.block {
        return Err("副系のサンプリング周波数とブロック長は主系と同じにしてください".into());
    }
    let a = Decoder::new(profile)?.run(frames_a, n, 0, cfg);
    let b = match frames_b {
        Some(fb) => Some(Decoder::new(pb)?.run(fb, n, offset_b, cfg)),
        None => None,
    };
    let (y, v) = match &b {
        None => (a.pcm.clone(), a.valid.clone()),
        Some(b) => fuse_band(&a, b, profile.block, profile.levels),
    };
    let plc_samples = v.iter().filter(|&&x| !x).count();
    let n_blocks = y.len().div_ceil(profile.block);
    let quality = block_quality(profile, &a.quality, b.as_ref().map(|b| b.quality.as_slice()), offset_b, n_blocks);
    let mut traces = vec![&a.rb];
    if let Some(b) = &b {
        traces.push(&b.rb);
    }
    let pcm = output_stage(profile, &y, &v, &quality, &traces);
    Ok(RxOutput { pcm, quality, stats_a: a.stats, stats_b: b.map(|b| b.stats), plc_samples })
}
