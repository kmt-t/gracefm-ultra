//! 受信（1 系統）: フレームのシンボル → 復号（decoder.rs）→ 補間（restore.rs、MendNet）→ 判定（gate.rs）→ 後段（post.rs、PostNet）→ 音声。
//! フレームを 1 つずつ入れ、音声は補間の窓が進むたび（HOP フレームごと）に出る。遅れは補間の確定待ちの DELAY フレーム（後段の先読みは暫定値で賄う）。

use crate::decoder::{Decoder, Stats};
use crate::gate::Gate;
use crate::post::{PostFilter, PostModel, AHEAD};
use crate::prior::PriorConfig;
use crate::profiles::Profile;
use crate::restore::{MendModel, Mended, Mender, Observation, HOP, NC, WIN};

// 補間の窓の、確定より後ろの暫定のフレームで、後段の先読みが足りること
const _: () = assert!(WIN - WIN / 4 - HOP >= AHEAD);

pub struct Receiver<M: MendModel, P: PostModel> {
    dec: Decoder,
    mender: Mender<M>,
    gate: Gate,
    post: PostFilter<P>,
    /// 超解像をかけたフレームの数と、判定の切り替えの回数
    pub post_on: usize,
    pub switches: usize,
}

impl<M: MendModel, P: PostModel> Receiver<M, P> {
    pub fn new(p: &Profile, cfg: PriorConfig, mend: M, post: P) -> Result<Receiver<M, P>, String> {
        if p.bits_s.iter().all(|&v| v == 0) || p.block != NC {
            return Err(format!("{}: MendNet と PostNet は M・S の 2 成分、ブロック長 64 のプロファイル用です", p.name));
        }
        let dec = Decoder::new(p, cfg)?;
        let mender = Mender::new(&dec.lay, mend);
        Ok(Receiver { dec, mender, gate: Gate::new(), post: PostFilter::new(post), post_on: 0, switches: 0 })
    }

    /// 次のフレームのシンボル（届かなければ None）→ 出せるようになった音声（L/R、満点 ±1。空か HOP × 64 サンプル前後）
    pub fn push(&mut self, sym: Option<&[[f64; 2]]>) -> Result<Vec<[f64; 2]>, String> {
        let d = self.dec.push(sym);
        let obs = Observation::new(&self.dec.lay, &d);
        match self.mender.push(obs)? {
            Some(m) => self.emit(m),
            None => Ok(Vec::new()),
        }
    }

    /// ストリームの終わり: 残りの音声
    pub fn finish(&mut self) -> Result<Vec<[f64; 2]>, String> {
        match self.mender.finish()? {
            Some(m) => self.emit(m),
            None => Ok(Vec::new()),
        }
    }

    pub fn stats(&self) -> &Stats {
        &self.dec.stats
    }

    fn emit(&mut self, m: Mended) -> Result<Vec<[f64; 2]>, String> {
        let fin = m
            .fin
            .into_iter()
            .map(|r| {
                let was = self.gate.is_on();
                let on = self.gate.push(&r);
                self.post_on += on as usize;
                self.switches += (on != was) as usize;
                (r, on)
            })
            .collect();
        self.post.push(fin, &m.ahead)
    }
}
