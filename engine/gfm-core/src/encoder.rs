//! 送信: PCM → フレームのシンボル。GraceFM `gracefm/txrx.py` の Transmitter の移植（Y 方式と暗号化は未対応）。
//! encode_frame = analyze → quantize → pack → scramble → modulate。デコーダ（decoder.rs）の鏡。

use crate::codec::{encode_block, BlockData};
use crate::framing::pack;
use crate::layout::Layout;
use crate::modem::Modem;
use crate::profiles::Profile;
use crate::scrambler::Scrambler;

pub struct Encoder {
    pub p: Profile,
    pub lay: Layout,
    pub modem: Modem,
    pub scrambler: Scrambler,
    pub nsym: usize,
    stream_id: u64,
    offset: usize,
    frame_no: u64,
    clip_opt: bool,
}

impl Encoder {
    /// stream_id: 系統（主 0、副 1）、offset: 先頭に入れる無音のサンプル数（デコーダの `run` の offset と対）
    pub fn new(p: &Profile, stream_id: u64, offset: usize) -> Result<Encoder, String> {
        let lay = Layout::new(p)?;
        let modem = Modem::new(p.qam16h, p.alpha);
        let nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
        let scrambler = Scrambler::new(nsym * modem.bits_per_symbol());
        Ok(Encoder { p: p.clone(), lay, modem, scrambler, nsym, stream_id, offset, frame_no: 0, clip_opt: true })
    }

    /// 1 ブロックのシンボル（プリアンブル込み）
    pub fn encode_frame(&mut self, pcm: &[[i64; 2]]) -> Vec<[i64; 2]> {
        self.encode_frame_parts(pcm).2
    }

    /// encode_frame と同じ。途中の BlockData と、スクランブル前の送信ビット列も返す（学習データ生成用）
    pub fn encode_frame_parts(&mut self, pcm: &[[i64; 2]]) -> (BlockData, Vec<u8>, Vec<[i64; 2]>) {
        let bd = encode_block(pcm, &self.p, self.clip_opt);
        let stream = pack(&self.lay, &bd, self.stream_id, self.frame_no);
        self.frame_no += 1;
        let mut buf = vec![0u8; self.nsym * self.modem.bits_per_symbol()];
        buf[..stream.len()].copy_from_slice(&stream);
        self.scrambler.scramble(&mut buf);
        let sym = self.modem.modulate(&buf, self.nsym);
        (bd, stream, sym)
    }

    /// 全体: 先頭に offset サンプルの無音を入れ、ブロックの整数倍に 0 で詰めて、フレームごとに符号化する
    pub fn run(&mut self, pcm: &[[i64; 2]]) -> Vec<Vec<[i64; 2]>> {
        let b = self.p.block;
        let mut x = vec![[0i64; 2]; self.offset];
        x.extend_from_slice(pcm);
        let nfr = (x.len() + b - 1) / b;
        x.resize(nfr * b, [0, 0]);
        (0..nfr).map(|i| self.encode_frame(&x[i * b..(i + 1) * b])).collect()
    }
}
