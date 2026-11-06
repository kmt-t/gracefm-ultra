//! 補間（MendNet）: 復号したフレームの列から、欠落・壊れたフレームの係数を埋め、ビット誤りを補正した係数（事後平均）と、その事後標準偏差を出す。
//! 窓 WIN フレームを HOP ずつずらして推論し、各窓の中央 HOP フレームを確定する。確定より後ろの窓の残りは、暫定値として後段の先読みに渡す。
//! 網の入力と出力の定義は `tools/export_onnx.py`（mendnet.onnx）。推論は `MendModel` に任せる。

use std::collections::VecDeque;

use crate::decoder::{block_steps, DecodedFrame, Status};
use crate::layout::Layout;

/// 窓 [フレーム]
pub const WIN: usize = 100;
/// 窓のずらし幅 [フレーム]。窓 [a, a + WIN) は [a + WIN/4, a + WIN/4 + HOP) を確定する（最初の窓は先頭から）
pub const HOP: usize = WIN / 2;
/// 確定までの遅れ [フレーム]: フレーム a + WIN/4 は、フレーム a + WIN − 1 が届いたときに確定する
pub const DELAY: usize = WIN - WIN / 4;
/// 成分（M、S）あたりの係数の数。成分は 2 つ
pub const NC: usize = 64;
/// 受信 LLR の量子化（網の入力の定義）: 自然対数の LLR の 4 倍を丸めて ±127 に飽和
const LLR_SCALE: f64 = 4.0;

/// 1 フレームの観測（網の入力の定義どおりに符号化した値）。係数の並びは、成分ごとに帯域 k = 0〜4 の係数を並べた 2 × 64
#[derive(Clone)]
pub struct Observation {
    /// 0 = 正常、1 = 高域欠落、2 = 壊れた（基本 CRC 不一致）、3 = 欠落
    pub status: u8,
    /// ヘッダ採用の、最尤との対数尤度の差 [nat]
    pub gap: f32,
    /// 復号した係数（満点 1.0。使えないフレームは 0）
    pub coef: Vec<f32>,
    /// (2, 帯域数) 誤差分散（情報なしは -1）
    pub var: Vec<f32>,
    /// 受信 LLR（量子化済み。仮数ビットの並び。欠落フレームは 0）
    pub llr: Vec<i8>,
    /// 復号したヘッダの刻み（満点 = 2^22 の尺度、不明は 0）
    pub dstep: Vec<f32>,
    /// 壊れたフレームの手がかり（`decoder::Hint`）。hint_h (2 × 64)、hint_q (2, 帯域数)。それ以外は 0
    pub hint_h: Vec<f32>,
    pub hint_q: Vec<f32>,
}

impl Observation {
    pub fn new(lay: &Layout, d: &DecodedFrame) -> Observation {
        let nb = lay.p.nbands();
        let status = match d.status {
            Status::Ok => 0,
            Status::HighFail => 1,
            Status::CrcFail => 2,
            Status::Lost => 3,
        };
        let mut llr = vec![0i8; lay.n_coef_bits()];
        for (q, &l) in llr.iter_mut().zip(&d.llr) {
            *q = (l * LLR_SCALE).round().clamp(-127.0, 127.0) as i8;
        }
        let mut dstep = vec![0.0; 2 * NC];
        if let Some((_, bd)) = &d.header {
            block_steps(lay, bd, d.status == Status::Ok, &mut dstep);
        }
        let (mut hint_h, mut hint_q) = (Vec::with_capacity(2 * NC), Vec::with_capacity(2 * nb));
        for ch in 0..2 {
            match &d.hint {
                Some(h) => {
                    hint_h.extend(h.h[ch].iter().map(|&v| v as f32));
                    hint_q.extend(h.q[ch][..nb].iter().map(|&v| v as f32));
                }
                None => {
                    hint_h.extend([0.0; NC]);
                    hint_q.extend(std::iter::repeat_n(0.0, nb));
                }
            }
        }
        Observation {
            status,
            gap: d.gap as f32,
            coef: d.coef.iter().flatten().flatten().map(|&v| v as f32).collect(),
            var: d.var.iter().flat_map(|v| v[..nb].iter().map(|&x| if x.is_finite() { x as f32 } else { -1.0 })).collect(),
            llr,
            dstep,
            hint_h,
            hint_q,
        }
    }

    /// 届かなかったフレーム（ストリームの末尾で窓を埋める）
    fn lost(nbit: usize, nb: usize) -> Observation {
        Observation {
            status: 3,
            gap: 0.0,
            coef: vec![0.0; 2 * NC],
            var: vec![-1.0; 2 * nb],
            llr: vec![0; nbit],
            dstep: vec![0.0; 2 * NC],
            hint_h: vec![0.0; 2 * NC],
            hint_q: vec![0.0; 2 * nb],
        }
    }

    /// 係数が使えるか（正常か高域欠落）
    pub fn usable(&self) -> bool {
        self.status <= 1
    }
}

/// 窓のバッチ（窓 × 成分）。並びは mendnet.onnx の入力と同じ: coef (b, WIN, 64)、var (b, WIN, nbands)、status (b, WIN)、gap (b, WIN)、comp (b)、
/// hint_h (b, WIN, 64)、hint_q (b, WIN, nbands)、llr (b, WIN, nb)（成分のビット順、詰めた分は 0）、dstep (b, WIN, 64)
pub struct MendBatch {
    pub b: usize,
    pub nb: usize,
    pub nbands: usize,
    pub coef: Vec<f32>,
    pub var: Vec<f32>,
    pub status: Vec<i64>,
    pub gap: Vec<f32>,
    pub comp: Vec<i64>,
    pub hint_h: Vec<f32>,
    pub hint_q: Vec<f32>,
    pub llr: Vec<f32>,
    pub dstep: Vec<f32>,
}

/// MendNet の推論
pub trait MendModel {
    /// 窓のバッチ → (e0、sd) 各 (b, WIN, 64)（満点 1.0）
    fn infer(&mut self, w: &MendBatch) -> Result<(Vec<f32>, Vec<f32>), String>;
}

/// 補間した 1 フレーム。e0、sd: M 64 + S 64（満点 1.0）
#[derive(Clone)]
pub struct Restored {
    pub e0: Vec<f32>,
    pub sd: Vec<f32>,
    /// 元のフレームの係数が使えたか（判定の統計に入れるか）
    pub usable: bool,
}

/// 1 つの窓の推論で出るフレーム
pub struct Mended {
    /// 確定したフレーム（前回の続きから）
    pub fin: Vec<Restored>,
    /// 確定より後ろの、窓の残り（暫定値。後段の先読みだけに使う）
    pub ahead: Vec<Restored>,
}

pub struct Mender<M: MendModel> {
    model: M,
    /// 成分ごとの、その成分の係数の仮数ビットの番号（観測の llr の中の位置）と、長い方の数
    bits: [Vec<usize>; 2],
    nb: usize,
    nbit: usize,
    nbands: usize,
    /// 直近 WIN フレームの観測
    buf: VecDeque<Observation>,
    received: usize,
    emitted: usize,
}

impl<M: MendModel> Mender<M> {
    pub fn new(lay: &Layout, model: M) -> Mender<M> {
        let bits = [0, 1].map(|ch| (0..lay.coef_flat.len()).filter(|&j| lay.coef_flat[j] / NC == ch).collect::<Vec<_>>());
        let nb = bits[0].len().max(bits[1].len());
        Mender {
            model,
            bits,
            nb,
            nbit: lay.n_coef_bits(),
            nbands: lay.p.nbands(),
            buf: VecDeque::with_capacity(WIN),
            received: 0,
            emitted: 0,
        }
    }

    /// 次のフレームの観測。窓が揃ったら推論して、確定したフレームと暫定のフレームを返す
    pub fn push(&mut self, obs: Observation) -> Result<Option<Mended>, String> {
        if self.buf.len() == WIN {
            self.buf.pop_front();
        }
        self.buf.push_back(obs);
        self.received += 1;
        if self.received < WIN || !(self.received - WIN).is_multiple_of(HOP) {
            return Ok(None);
        }
        let out = self.run(WIN / 4 + HOP)?;
        Ok(Some(out))
    }

    /// ストリームの終わり: 確定していないフレームを、最後の窓（足りなければ欠落で埋める）で確定する
    pub fn finish(&mut self) -> Result<Option<Mended>, String> {
        if self.emitted == self.received {
            return Ok(None);
        }
        let real = self.buf.len();
        while self.buf.len() < WIN {
            self.buf.push_back(Observation::lost(self.nbit, self.nbands));
        }
        let mut out = self.run(real)?;
        out.ahead.clear();
        Ok(Some(out))
    }

    /// 今の窓（buf）を推論して、窓の中の位置 end までを確定する
    fn run(&mut self, end: usize) -> Result<Mended, String> {
        let a = self.received.saturating_sub(WIN);
        let (e0, sd) = self.model.infer(&self.batch())?;
        let frame = |t: usize| {
            let pick = |x: &[f32]| (0..2).flat_map(|ch| x[(ch * WIN + t) * NC..(ch * WIN + t + 1) * NC].iter().copied()).collect();
            Restored { e0: pick(&e0), sd: pick(&sd), usable: self.buf[t].usable() }
        };
        let lo = self.emitted - a;
        let fin = (lo..end).map(frame).collect();
        let ahead = (end..WIN).map(frame).collect();
        self.emitted = a + end;
        Ok(Mended { fin, ahead })
    }

    /// 今の窓を、成分ごとに 1 つずつ並べる（M、S）
    fn batch(&self) -> MendBatch {
        let (nb, k) = (self.nb, self.nbands);
        let mut w = MendBatch {
            b: 2,
            nb,
            nbands: k,
            coef: Vec::with_capacity(2 * WIN * NC),
            var: Vec::with_capacity(2 * WIN * k),
            status: Vec::with_capacity(2 * WIN),
            gap: Vec::with_capacity(2 * WIN),
            comp: vec![0, 1],
            hint_h: Vec::with_capacity(2 * WIN * NC),
            hint_q: Vec::with_capacity(2 * WIN * k),
            llr: vec![0.0; 2 * WIN * nb],
            dstep: Vec::with_capacity(2 * WIN * NC),
        };
        for ch in 0..2 {
            let (c, v) = (ch * NC..(ch + 1) * NC, ch * k..(ch + 1) * k);
            for (t, o) in self.buf.iter().enumerate() {
                w.coef.extend_from_slice(&o.coef[c.clone()]);
                w.hint_h.extend_from_slice(&o.hint_h[c.clone()]);
                w.dstep.extend_from_slice(&o.dstep[c.clone()]);
                w.var.extend_from_slice(&o.var[v.clone()]);
                w.hint_q.extend_from_slice(&o.hint_q[v.clone()]);
                w.status.push(o.status as i64);
                w.gap.push(o.gap);
                let row = &mut w.llr[(ch * WIN + t) * nb..(ch * WIN + t) * nb + self.bits[ch].len()];
                for (x, &j) in row.iter_mut().zip(&self.bits[ch]) {
                    *x = o.llr[j] as f32;
                }
            }
        }
        w
    }
}
