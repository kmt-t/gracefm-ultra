//! 受信: フレームのシンボル → PCM ブロック。GraceFM `gracefm/txrx.py` の StreamDecoder の移植（Y 方式と暗号化は未対応）。
//! decode_frame = demodulate → descramble → unpack（Chase 救済つき）→ dequantize（軟判定）→ synthesize。エンコーダ（encoder.rs）の鏡。
//! 結果は戻り値で返す（品質コードや誤差分散を内部状態に隠さない）。

use crate::codec::quant::{dequantize_soft, Rb, INF_LOG};
use crate::codec::synthesize;
use crate::dsp::sigmoid_q15;
use crate::framing::{unpack, unpack_recovering_with, Unpacked};
use crate::layout::Layout;
use crate::modem::{Modem, N_PREAMBLE};
use crate::profiles::Profile;
use crate::scrambler::Scrambler;

/// 品質コード: 0 正常、1 高域欠落、2 欠落
pub const Q_OK: i64 = 0;
pub const Q_NO_HIGH: i64 = 1;
pub const Q_LOST: i64 = 2;

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub frames: usize,
    pub lost: usize,
    pub crc_fail: usize,
    pub high_crc_fail: usize,
    pub recovered: usize,
}

pub struct DecodedFrame {
    pub pcm: Vec<[i64; 2]>,
    pub valid: bool,
    pub quality: i64,
    /// 半ブロック・成分・帯域ごとの誤差分散（対数領域）。届かなければ None
    pub rb: Option<[Rb; 2]>,
}

/// 系統全体の復号結果
pub struct StreamOutput {
    pub pcm: Vec<[i64; 2]>,
    pub valid: Vec<bool>,
    /// フレームごとの品質コード
    pub quality: Vec<i64>,
    /// サンプルごとの誤差分散（対数領域）[成分][帯域]
    pub rb_trace: Vec<Rb>,
}

pub struct Decoder {
    pub p: Profile,
    pub lay: Layout,
    modem: Modem,
    scrambler: Scrambler,
    digital_nsym: usize,
    high_absent: bool,
    pub stats: Stats,
}

impl Decoder {
    pub fn new(p: &Profile) -> Result<Decoder, String> {
        let lay = Layout::new(p)?;
        let modem = Modem::new(p.qam16h, p.alpha);
        let digital_nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
        let scrambler = Scrambler::new(digital_nsym * modem.bits_per_symbol());
        let high_absent = p.bits_m[p.high_k()..].iter().all(|&b| b == 0) && p.bits_s[p.high_k()..].iter().all(|&b| b == 0);
        Ok(Decoder { p: p.clone(), lay, modem, scrambler, digital_nsym, high_absent, stats: Stats::default() })
    }

    /// Chase 救済つきの unpack。返り値の 2 つ目は反転した個数
    fn unpack_recovering(&self, soft: &[i64], bits: &[u8]) -> Option<(Unpacked, usize)> {
        unpack_recovering_with(&self.lay, soft, bits, &|v| -v, &|v: i64| v.abs() as f64, &|s, b| unpack(&self.lay, s, b))
    }

    fn lost_frame(&self) -> DecodedFrame {
        DecodedFrame { pcm: vec![[0, 0]; self.p.block], valid: false, quality: Q_LOST, rb: None }
    }

    /// 1 フレームの受信シンボル（プリアンブル込み。届かなければ None）→ 復号結果
    pub fn decode_frame(&mut self, sym: Option<&[[i64; 2]]>) -> DecodedFrame {
        self.stats.frames += 1;
        let expected = N_PREAMBLE + self.digital_nsym;
        let sym = match sym {
            Some(s) if s.len() == expected => s,
            _ => {
                self.stats.lost += 1;
                return self.lost_frame();
            }
        };
        let digital = &sym[N_PREAMBLE..expected];
        let (mut llr, mut hard, _var) = self.modem.demodulate(digital, self.lay.n_tx);
        self.scrambler.descramble(&mut llr, &mut hard);
        let (u, recovered) = match self.unpack_recovering(&llr, &hard) {
            Some(r) => r,
            None => {
                self.stats.crc_fail += 1;
                return self.lost_frame();
            }
        };
        if recovered > 0 {
            self.stats.recovered += 1;
        }
        let mut p1: Vec<i64> = self.lay.coef_stream_idx.iter().map(|&i| sigmoid_q15(llr[i])).collect();
        for (t, &ci) in self.lay.top_coef_idx.iter().enumerate() {
            p1[ci] = u.protected_bits[t] as i64 * 32768; // FEC で訂正した保護ビットは確定値
        }
        let quality = if u.high_ok && !self.high_absent { Q_OK } else { Q_NO_HIGH };
        if !u.high_ok {
            self.stats.high_crc_fail += 1;
        }
        let est = dequantize_soft(&u.bd, &self.lay, &p1, u.high_ok);
        DecodedFrame { pcm: synthesize(&est.coef), valid: true, quality, rb: Some(est.rb) }
    }

    /// frames: フレームごとのシンボル（届かなければ None）。返り値は、offset サンプル目から n サンプル
    pub fn run(&mut self, frames: &[Option<Vec<[i64; 2]>>], n: usize, offset: usize) -> StreamOutput {
        let b = self.p.block;
        let h = b / 2;
        let nf = frames.len();
        let mut pcm = vec![[0i64; 2]; nf * b];
        let mut valid = vec![false; nf * b];
        let mut rb = vec![[[INF_LOG; 5]; 2]; nf * b];
        let mut quality = vec![0; nf];
        for (i, f) in frames.iter().enumerate() {
            let d = self.decode_frame(f.as_deref());
            pcm[i * b..(i + 1) * b].copy_from_slice(&d.pcm);
            for t in 0..b {
                valid[i * b + t] = d.valid;
            }
            quality[i] = d.quality;
            if let Some(r) = d.rb {
                for t in 0..h {
                    rb[i * b + t] = r[0];
                    rb[i * b + h + t] = r[1];
                }
            }
        }
        let end = (offset + n).min(pcm.len());
        let start = offset.min(end);
        StreamOutput { pcm: pcm[start..end].to_vec(), valid: valid[start..end].to_vec(), quality, rb_trace: rb[start..end].to_vec() }
    }
}
