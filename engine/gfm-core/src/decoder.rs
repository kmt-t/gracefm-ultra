//! 受信: フレームのシンボル → PCM ブロック（浮動小数点）。GraceFM `gracefm/decoder.py` に相当（Y 方式と暗号化は未対応）。
//! decode_frame = demodulate → descramble → unpack（Chase 救済つき）→ soft_coeffs（軟判定の逆量子化）→ synthesize。エンコーダ（encoder.rs）の鏡。
//! 結果は戻り値で返す（品質コードや誤差分散を内部状態に隠さない）。信号の尺度は満点 ±1。

use crate::codec::quant::{exp_range, qmax_of, qmin_of};
use crate::codec::synthesize;
use crate::codec::BlockData;
use crate::framing::{coefficient_bits, unpack_recovering};
use crate::layout::Layout;
use crate::modem::{Modem, N_PREAMBLE};
use crate::profiles::{Profile, MAX_BANDS};
use crate::scrambler::Scrambler;

/// 品質コード: 0 正常、1 高域欠落、2 欠落
pub const QUALITY_NORMAL: i64 = 0;
pub const QUALITY_HIGH_MISSING: i64 = 1;
pub const QUALITY_MISSING: i64 = 2;

/// フレームの状態
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Ok,
    /// 高域 CRC だけ不一致
    HighFail,
    /// 届かない
    Lost,
    /// 基本 CRC が不一致
    CrcFail,
}

/// 誤差分散（線形、満点 1.0 に対する二乗）: [成分][帯域]。情報なしは f64::INFINITY
pub type Rb = [[f64; MAX_BANDS]; 2];

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub frames: usize,
    pub lost: usize,
    pub crc_fail: usize,
    pub high_crc_fail: usize,
    pub recovered: usize,
}

pub struct DecodedFrame {
    /// (block) の L/R（満点 ±1）
    pub pcm: Vec<[f64; 2]>,
    pub status: Status,
    pub quality: i64,
    /// 半ブロックごと・成分・帯域の誤差分散。届かない・CRC 不一致は None
    pub rb: Option<[Rb; 2]>,
}

impl DecodedFrame {
    /// 基本 CRC が通ったか（高域 CRC の結果は問わない）
    pub fn valid(&self) -> bool {
        matches!(self.status, Status::Ok | Status::HighFail)
    }
}

/// 系統全体の復号結果
pub struct StreamResult {
    /// 時刻合わせ済みの音声
    pub pcm: Vec<[f64; 2]>,
    /// 時刻合わせ済みの有効フラグ
    pub valid: Vec<bool>,
    /// フレームごとの品質コード（時刻合わせ前）
    pub quality: Vec<i64>,
    /// 時刻合わせ済みの、サンプルごとの誤差分散 [成分][帯域]
    pub rb: Vec<Rb>,
    pub stats: Stats,
}

/// 軟判定の逆量子化。p1: 仮数の各ビットが 1 である確率（coef_items の並び）。
/// 期待値は 仮数 + Σ (p1 − 硬判定ビット)·2^pos、誤差分散は (Σ p1(1−p1)·4^pos + 1/12)·刻み²。
/// high_ok が false のときは、高域（high_k 以上の帯域）を 0（情報なし）にする。返り値: (帯域係数 [ch][k][j]、半ブロックごとの誤差分散)
pub fn soft_coeffs(lay: &Layout, bd: &BlockData, p1: &[f64], high_ok: bool) -> (Vec<Vec<Vec<f64>>>, [Rb; 2]) {
    let p = &lay.p;
    let nb = p.nbands();
    let mut du = vec![0f64; lay.n_coef];
    let mut var = vec![0f64; lay.n_coef];
    for (j, &cf) in lay.coef_flat.iter().enumerate() {
        let hard = if p1[j] > 0.5 { 1.0 } else { 0.0 };
        let w = (1u64 << lay.coef_pos[j]) as f64;
        du[cf] += (p1[j] - hard) * w;
        var[cf] += p1[j] * (1.0 - p1[j]) * w * w;
    }
    let mut coef = vec![vec![Vec::new(); nb]; 2];
    let mut rb = [[[f64::INFINITY; MAX_BANDS]; 2]; 2];
    for ch in 0..2 {
        for k in 0..nb {
            let b = p.bits(ch)[k];
            let n = p.ncoef(k);
            if b == 0 || (k >= p.high_k() && !high_ok) {
                coef[ch][k] = vec![0.0; n];
                continue;
            }
            let o = lay.coef_offset[&(ch, k)];
            let (qmax, qmin) = (qmax_of(b) as f64, qmin_of(b) as f64);
            let mant = bd.mant[ch][k].as_ref().unwrap();
            let attn = bd.attn[ch][k].as_ref().unwrap();
            let e = bd.eidx[ch][k].unwrap() + exp_range(k, b, p).0; // 刻みの指数（満点 = 1.0）
            let mut c = Vec::with_capacity(n);
            let mut rv = Vec::with_capacity(n);
            for j in 0..n {
                let m = (mant[j] as f64 + du[o + j]).clamp(qmin, qmax);
                let a = p.attn_at(ch, k, attn, j);
                let step = 2f64.powi((e - a) as i32);
                c.push(m * step);
                rv.push((var[o + j] + 1.0 / 12.0) * step * step);
            }
            let halves = if n >= 2 {
                let h = (n / 2) as f64;
                [rv[..n / 2].iter().sum::<f64>() / h, rv[n / 2..].iter().sum::<f64>() / h]
            } else {
                [rv[0], rv[0]]
            };
            for hh in 0..2 {
                rb[hh][ch][k] = halves[hh];
            }
            coef[ch][k] = c;
        }
    }
    (coef, rb)
}

/// BlockData → 帯域係数を、通信路なしで復元する。硬判定の復号は軟判定の特殊ケース（各仮数ビットの確率を 0 か 1 にして soft_coeffs に通す）
pub fn reconstruct_coeffs(lay: &Layout, bd: &BlockData) -> Vec<Vec<Vec<f64>>> {
    let p1: Vec<f64> = coefficient_bits(lay, bd).iter().map(|&b| b as f64).collect();
    soft_coeffs(lay, bd, &p1, true).0
}

/// BlockData → PCM ブロックを、通信路なしで復元する（encode_block の逆）
pub fn reconstruct(lay: &Layout, bd: &BlockData) -> Vec<[f64; 2]> {
    synthesize(&reconstruct_coeffs(lay, bd))
}

pub struct Decoder {
    pub p: Profile,
    pub lay: Layout,
    modem: Modem,
    scrambler: Scrambler,
    digital_nsym: usize,
    high_absent: bool,
}

impl Decoder {
    pub fn new(p: &Profile) -> Result<Decoder, String> {
        let lay = Layout::new(p)?;
        let modem = Modem::new(p.qam16h, p.alpha);
        let digital_nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
        let scrambler = Scrambler::new(digital_nsym * modem.bits_per_symbol());
        let high_absent = p.bits_m[p.high_k()..].iter().all(|&b| b == 0) && p.bits_s[p.high_k()..].iter().all(|&b| b == 0);
        Ok(Decoder { p: p.clone(), lay, modem, scrambler, digital_nsym, high_absent })
    }

    fn missing(&self, status: Status) -> DecodedFrame {
        DecodedFrame { pcm: vec![[0.0, 0.0]; self.p.block], status, quality: QUALITY_MISSING, rb: None }
    }

    /// 1 フレームの受信シンボル（プリアンブル込み、±1 = 1.0。届かなければ None）→ 復号結果。返り値の 2 つ目は Chase 救済で反転した bit の数
    pub fn decode_frame(&self, sym: Option<&[[f64; 2]]>) -> (DecodedFrame, usize) {
        let expected = N_PREAMBLE + self.digital_nsym;
        let sym = match sym {
            Some(s) if s.len() == expected => s,
            _ => return (self.missing(Status::Lost), 0),
        };
        let (mut llr, mut hard, _var) = self.modem.demodulate(&sym[N_PREAMBLE..expected], self.lay.n_tx);
        self.scrambler.descramble(&mut llr, &mut hard);
        let lay = &self.lay;
        let (u, recovered) = match unpack_recovering(lay, &llr, &hard) {
            Some(r) => r,
            None => return (self.missing(Status::CrcFail), 0),
        };
        // 1 である確率 = 1 / (1 + e^LLR)（LLR が正なら 0 らしい）
        let mut p1: Vec<f64> = lay.coef_stream_idx.iter().map(|&i| 1.0 / (1.0 + llr[i].clamp(-40.0, 40.0).exp())).collect();
        for (t, &ci) in lay.top_coef_idx.iter().enumerate() {
            p1[ci] = u.top_bits[t] as f64; // FEC で訂正した保護ビットは確定値
        }
        let (status, quality) = if u.high_ok {
            (Status::Ok, if self.high_absent { QUALITY_HIGH_MISSING } else { QUALITY_NORMAL })
        } else {
            (Status::HighFail, QUALITY_HIGH_MISSING)
        };
        let (coef, rb) = soft_coeffs(lay, &u.bd, &p1, u.high_ok);
        (DecodedFrame { pcm: synthesize(&coef), status, quality, rb: Some(rb) }, recovered)
    }

    /// frames: フレームごとのシンボル（届かなければ None）。返り値は、offset サンプル目から n サンプル
    pub fn run(&self, frames: &[Option<Vec<[f64; 2]>>], n: usize, offset: usize) -> StreamResult {
        let b = self.p.block;
        let h = b / 2;
        let nf = frames.len();
        let mut pcm = vec![[0f64; 2]; nf * b];
        let mut valid = vec![false; nf * b];
        let mut rb = vec![[[f64::INFINITY; MAX_BANDS]; 2]; nf * b];
        let mut quality = vec![0; nf];
        let mut stats = Stats { frames: nf, ..Stats::default() };
        for (i, f) in frames.iter().enumerate() {
            let (d, recovered) = self.decode_frame(f.as_deref());
            pcm[i * b..(i + 1) * b].copy_from_slice(&d.pcm);
            for t in 0..b {
                valid[i * b + t] = d.valid();
            }
            quality[i] = d.quality;
            if let Some(r) = d.rb {
                for t in 0..h {
                    rb[i * b + t] = r[0];
                    rb[i * b + h + t] = r[1];
                }
            }
            match d.status {
                Status::Lost => stats.lost += 1,
                Status::CrcFail => stats.crc_fail += 1,
                Status::HighFail => stats.high_crc_fail += 1,
                Status::Ok => {}
            }
            stats.recovered += (recovered > 0) as usize;
        }
        let end = (offset + n).min(pcm.len());
        let start = offset.min(end);
        StreamResult { pcm: pcm[start..end].to_vec(), valid: valid[start..end].to_vec(), quality, rb: rb[start..end].to_vec(), stats }
    }
}
