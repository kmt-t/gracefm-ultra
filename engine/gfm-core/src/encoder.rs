//! 送信: PCM → フレームのシンボル（浮動小数点）。GraceFM `gracefm/encoder.py` に相当（暗号化は未対応）。
//! analyze_stream → フレームごとに quantize → pack → scramble → modulate。デコーダ（decoder.rs）の鏡。
//! ビット配置・FEC・CRC・スクランブルはビット処理なので、本家とビット単位で同じ。音声の変換と量子化だけが浮動小数点。

use crate::codec::{analyze_stream, quantize, BlockData};
use crate::fec;
use crate::framing::{pack, y_msb, y_u};
use crate::layout::Layout;
use crate::modem::{prbs, y_modulate, Modem};
use crate::profiles::Profile;
use crate::scrambler::Scrambler;

/// ヘッダの系統のビット（主系）
const MAIN_STREAM: u64 = 0;

pub struct Encoder {
    pub p: Profile,
    pub lay: Layout,
    pub modem: Modem,
    pub scrambler: Scrambler,
    /// デジタル変調のシンボル数（Y の専用シンボルはその後ろ）
    pub nsym: usize,
    y_mask: Vec<u8>,
    frame_no: u64,
    clip_opt: bool,
}

impl Encoder {
    pub fn new(p: &Profile) -> Result<Encoder, String> {
        let lay = Layout::new(p)?;
        let modem = Modem::new(p.qam16h, p.alpha);
        let nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
        let digital_bits = nsym * modem.bits_per_symbol();
        let scrambler = Scrambler::new(digital_bits);
        let y_mask = prbs(digital_bits + 3 * lay.y_items.len()).split_off(digital_bits);
        Ok(Encoder { p: p.clone(), lay, modem, scrambler, nsym, y_mask, frame_no: 0, clip_opt: true })
    }

    /// 1 フレームの帯域係数のシンボル（プリアンブル込み）
    pub fn frame(&mut self, bands: &[Vec<Vec<f64>>]) -> Vec<[f64; 2]> {
        self.frame_parts(bands).2
    }

    /// frame と同じ。途中の BlockData と、スクランブル前の送信ビット列も返す（学習データ生成用）
    pub fn frame_parts(&mut self, bands: &[Vec<Vec<f64>>]) -> (BlockData, Vec<u8>, Vec<[f64; 2]>) {
        let bd = quantize(bands, &self.p, self.clip_opt);
        let stream = pack(&self.lay, &bd, MAIN_STREAM, self.frame_no);
        self.frame_no += 1;
        let mut buf = vec![0u8; self.nsym * self.modem.bits_per_symbol()];
        buf[..stream.len()].copy_from_slice(&stream);
        self.scrambler.scramble(&mut buf);
        let mut sym = self.modem.modulate(&buf, self.nsym);
        if !self.lay.y_items.is_empty() {
            let u = y_u(&self.lay, &bd);
            let msb = y_msb(&self.lay, &u);
            sym.extend(y_modulate(&u, &fec::rsc_parity(&msb).0[..u.len()], &self.y_mask));
        }
        (bd, stream, sym)
    }

    /// 全体: ブロックの整数倍に 0 で詰めて、フレームごとに符号化する
    pub fn run(&mut self, pcm: &[[f64; 2]]) -> Vec<Vec<[f64; 2]>> {
        let b = self.p.block;
        let mut x = pcm.to_vec();
        x.resize(x.len().div_ceil(b) * b, [0.0, 0.0]);
        analyze_stream(&x, &self.p).iter().map(|bands| self.frame(bands)).collect()
    }
}
