//! 後段（PostNet、MDCT 領域の超解像）: 補間した係数 → 音声。判定がオフの MDCT フレームは MendNet の係数のまま（網の中で切り替える）。
//! PostNet は段ごとに過去の入力を持ち越して逐次に回す（`tools/export_onnx.py` の postnet.onnx）。MDCT のフレーム f は波形 [256(f−1), 256(f+1))、
//! フレーム [4f − 4, 4f + 4) にかかる。波形の合成には前後 REACH フレームの係数も要る。PostNet の出力のフレーム f は、MDCT のフレーム f + LOOKAHEAD までを見る。
//! 確定したフレームだけで決まる位置まで状態を進め、その先（暫定のフレームに頼る部分）は毎回計算し直す。
//! フレームの末尾の標本には次のフレームの係数が届くので、音声は確定したフレームより REACH フレーム手前まで出す。
//! 全体を一度に通したのと同じ出力になる（先読みが暫定値のぶんを除く）。

use std::collections::VecDeque;

use crate::restore::{Restored, NC};

/// MDCT のホップ [フレーム]（256 サンプル）
pub const MDCT_HOP: usize = 4;
/// MDCT のビン数
pub const BINS: usize = MDCT_HOP * NC;
/// PostNet の先読み [MDCT フレーム]
pub const LOOKAHEAD: usize = 2;
/// 先読みに使う暫定のフレームの最大数 [フレーム]
pub const AHEAD: usize = MDCT_HOP * (LOOKAHEAD + 2) - 1 + REACH;
/// フレームの係数が届く隣のフレーム数（前 15、後 16 サンプル）
pub const REACH: usize = 1;

/// PostNet の 1 歩の入力。MDCT のフレーム [k, k + frames) を処理して、PostNet の出力 [k, k + frames − LOOKAHEAD) を出す
pub struct PostInput<'a> {
    /// (2, 4·frames + 4 + 2·REACH, 64) M/S のフレーム [4k − 4 − REACH, 4k + 4·frames + REACH)
    pub e0: &'a [f32],
    pub sd: &'a [f32],
    pub frames: usize,
    /// (frames − LOOKAHEAD) MDCT のフレームごとの判定（1 = 超解像）
    pub on: &'a [f32],
    /// (2, BINS) 切り替え後の MDCT のフレーム k − 1
    pub yprev: &'a [f32],
    /// 段ごとの状態 (2, チャンネル, 幅): 段の入力の、フレーム [k − 幅, k)
    pub cache: &'a [Vec<f32>],
}

pub struct PostOutput {
    /// (2, BINS·L) M/S の音声 [256(k − 1), 256(k + L − 1))。L = frames − LOOKAHEAD
    pub audio: Vec<f32>,
    /// (2, BINS, L) 切り替え後の MDCT のフレーム [k, k + L)
    pub y: Vec<f32>,
    /// 段ごとの入力 (2, チャンネル, 幅 + この歩の長さ)。フレーム k − 幅 から
    pub u: Vec<Vec<f32>>,
}

/// PostNet の推論
pub trait PostModel {
    /// 段ごとの状態の (チャンネル数, 幅)
    fn state_dims(&self) -> Vec<(usize, usize)>;
    fn step(&mut self, x: &PostInput) -> Result<PostOutput, String>;
}

pub struct PostFilter<P: PostModel> {
    model: P,
    dims: Vec<(usize, usize)>,
    cache: Vec<Vec<f32>>,
    yprev: Vec<f32>,
    /// 状態の位置（MDCT のフレーム）
    k: usize,
    /// 確定したフレームと判定。先頭はフレーム start（= max(4k − 4 − REACH, 0)）
    hist: VecDeque<(Restored, bool)>,
    start: usize,
    /// 音声にした フレームの数
    done: usize,
}

impl<P: PostModel> PostFilter<P> {
    pub fn new(model: P) -> PostFilter<P> {
        let dims = model.state_dims();
        let cache = dims.iter().map(|&(ch, w)| vec![0.0; 2 * ch * w]).collect();
        PostFilter { model, dims, cache, yprev: vec![0.0; 2 * BINS], k: 0, hist: VecDeque::new(), start: 0, done: 0 }
    }

    /// 確定したフレーム fin（判定つき）を足して、出せるようになった音声（L/R、満点 ±1）を返す。ahead: その後ろの暫定のフレーム（判定は確定の最後と同じとする）。
    /// ストリームの終わりは ahead を空にする（後ろは無音で、残りの音声を全部出す）
    pub fn push(&mut self, fin: Vec<(Restored, bool)>, ahead: &[Restored]) -> Result<Vec<[f64; 2]>, String> {
        self.hist.extend(fin);
        let (k, c) = (self.k, self.start + self.hist.len());
        let to = if ahead.is_empty() { c } else { c.saturating_sub(REACH) };
        if to <= self.done {
            return Ok(Vec::new());
        }
        let last_on = self.hist.back().is_none_or(|h| h.1);
        // 音声 [64·done, 64·to) を出すのに要る MDCT のフレーム [k, e)
        let e = to.div_ceil(MDCT_HOP) + LOOKAHEAD + 1;
        let (f, l) = (e - k, e - k - LOOKAHEAD);
        let n = MDCT_HOP * (f + 1) + 2 * REACH;
        let (mut e0, mut sd) = (vec![0f32; 2 * n * NC], vec![0f32; 2 * n * NC]);
        // フレームの判定。ストリームの前後はオンとみなす（全体を一度に通すときと同じ）
        let mut gate = vec![true; n];
        for (i, g) in gate.iter_mut().enumerate() {
            let Some(h) = (MDCT_HOP * k + i).checked_sub(MDCT_HOP + REACH) else { continue };
            let r = if h < c {
                let (r, on) = &self.hist[h - self.start];
                *g = *on;
                Some(r)
            } else {
                *g = ahead.is_empty() || last_on;
                ahead.get(h - c)
            };
            if let Some(r) = r {
                for ch in 0..2 {
                    let dst = (ch * n + i) * NC..(ch * n + i + 1) * NC;
                    e0[dst.clone()].copy_from_slice(&r.e0[ch * NC..(ch + 1) * NC]);
                    sd[dst].copy_from_slice(&r.sd[ch * NC..(ch + 1) * NC]);
                }
            }
        }
        // MDCT のフレーム k + j の判定: かかる フレーム（この歩の [REACH + 4j, REACH + 4j + 8)）が全部オン
        let on: Vec<f32> = (0..l).map(|j| gate[REACH + MDCT_HOP * j..REACH + MDCT_HOP * (j + 2)].iter().all(|&x| x) as u8 as f32).collect();
        let out = self.model.step(&PostInput { e0: &e0, sd: &sd, frames: f, on: &on, yprev: &self.yprev, cache: &self.cache })?;
        // 音声 [256(k − 1), …) から、[64·done, 64·to) を採る
        let (len, off) = (l * BINS, BINS * k);
        let pcm = (NC * self.done + BINS - off..NC * to + BINS - off)
            .map(|i| [(out.audio[i] + out.audio[len + i]) as f64, (out.audio[i] - out.audio[len + i]) as f64])
            .collect();
        self.commit(c, &out);
        self.done = to;
        Ok(pcm)
    }

    /// 確定した フレーム c だけで決まる位置まで、状態を進める
    fn commit(&mut self, c: usize, out: &PostOutput) {
        let k = self.k;
        let k2 = (c / MDCT_HOP).saturating_sub(LOOKAHEAD).max(k);
        for (i, &(ch, w)) in self.dims.iter().enumerate() {
            let total = out.u[i].len() / (2 * ch);
            for row in 0..2 * ch {
                let src = row * total + k2 - k;
                self.cache[i][row * w..(row + 1) * w].copy_from_slice(&out.u[i][src..src + w]);
            }
        }
        if k2 > k {
            let l = out.y.len() / (2 * BINS);
            for (row, y) in self.yprev.iter_mut().enumerate() {
                *y = out.y[row * l + k2 - k - 1];
            }
        }
        let start = (MDCT_HOP * k2).saturating_sub(MDCT_HOP + REACH);
        self.hist.drain(..start - self.start);
        self.start = start;
        self.k = k2;
    }
}
