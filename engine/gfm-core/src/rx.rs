//! 受信（1 系統のデコーダと、2 系統の受信）。GraceFM `gracefm/txrx.py` の StreamDecoder と receive の移植。Y 方式と暗号化は未対応。

use crate::codec::{exp_range, haar_inverse, ms_to_lr, qmax_of, qmin_of, BlockData, FRAC};
use crate::dsp::{log2_q8, saturate, shift_signed, sigmoid_q15};
use crate::framing::unpack_recovering;
use crate::fusion::{fuse_band, INF_LOG};
use crate::iir::apply_alt;
use crate::layout::Layout;
use crate::modem::{prbs, Modem, N_PREAMBLE};
use crate::plc::{conceal, crossfade_switch, PlcParams};
use crate::profiles::Profile;

const ONE_TWELFTH_Q15: i64 = 2731;

pub type Rb = [[i64; 5]; 2]; // [ci][k]

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub frames: usize,
    pub lost: usize,
    pub crc_fail: usize,
    pub high_crc_fail: usize,
    pub recovered: usize,
}

pub struct StreamDecoder {
    pub p: Profile,
    pub lay: Layout,
    modem: Modem,
    digital_nsym: usize,
    high_absent: bool,
    scr: Vec<u8>,
    pub stats: Stats,
    pub quality: Vec<i64>,
    pub rb_trace: Vec<Rb>,
}

impl StreamDecoder {
    pub fn new(p: &Profile) -> Result<StreamDecoder, String> {
        let lay = Layout::new(p)?;
        let modem = Modem::new(p.qam16h, p.alpha);
        let digital_nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
        let high_absent = p.bits_m[p.high_k()..].iter().all(|&b| b == 0) && p.bits_s[p.high_k()..].iter().all(|&b| b == 0);
        let scr = prbs(lay.n_tx);
        Ok(StreamDecoder { p: p.clone(), lay, modem, digital_nsym, high_absent, scr, stats: Stats::default(), quality: Vec::new(), rb_trace: Vec::new() })
    }

    /// 軟判定の復号。rb は半ブロック・M/S・帯域の誤差分散（対数領域）
    fn soft_coeffs(&self, bd: &BlockData, p1: &[i64], high_ok: bool) -> (Vec<Vec<Vec<i64>>>, [Rb; 2]) {
        let (p, lay) = (&self.p, &self.lay);
        let nb = p.nbands();
        let mut du = vec![0i64; lay.n_coef];
        let mut var = vec![0i64; lay.n_coef];
        for (j, &cf) in lay.coef_flat.iter().enumerate() {
            let hard = (p1[j] > 16384) as i64;
            let w = 1i64 << lay.coef_pos[j];
            du[cf] += (p1[j] - hard * 32768) * w;
            var[cf] += ((p1[j] * (32768 - p1[j])) >> 15) * w * w;
        }
        let mut coefs = vec![vec![Vec::new(); nb]; 2];
        let mut rb = [[[INF_LOG; 5]; 2]; 2];
        for ch in 0..2 {
            for k in 0..nb {
                let b = p.bits(ch)[k];
                let n = p.ncoef(k);
                if b == 0 || (k >= p.high_k() && !high_ok) {
                    coefs[ch][k] = vec![0; n];
                    continue;
                }
                let o = lay.coef_offset[&(ch, k)];
                let (qmax, qmin) = (qmax_of(b), qmin_of(b));
                let mant = bd.mant[ch][k].as_ref().unwrap();
                let attn = bd.attn[ch][k].as_ref().unwrap();
                let group = p.attenuator_group(ch, k);
                let eidx = bd.eidx[ch][k].unwrap();
                let base = eidx + exp_range(k, b, p).0 + FRAC;
                let mut c = Vec::with_capacity(n);
                let mut rv = Vec::with_capacity(n);
                for j in 0..n {
                    let m_q15 = ((mant[j] << 15) + du[o + j]).clamp(qmin << 15, qmax << 15);
                    let a = if group > 0 { attn[j / group] } else { 0 };
                    let s = base - a;
                    c.push(saturate(shift_signed(m_q15, s - 15)));
                    rv.push((var[o + j] + ONE_TWELFTH_Q15) >> (2 * a));
                }
                let exp_two = 2 * base - 15;
                let halves = if n >= 2 {
                    let h = (n / 2) as i64;
                    (rv[..n / 2].iter().sum::<i64>() / h, rv[n / 2..].iter().sum::<i64>() / h)
                } else {
                    (rv[0], rv[0])
                };
                let hv = [halves.0, halves.1];
                for hh in 0..2 {
                    rb[hh][ch][k] = log2_q8(hv[hh].max(1)) + (exp_two << 8);
                }
                coefs[ch][k] = c;
            }
        }
        (coefs, rb)
    }

    /// 1 フレームの受信シンボル（プリアンブル込み。届かなければ None）→（ブロック, 有効か, 品質, rb）
    pub fn decode(&mut self, sym: Option<&[[i64; 2]]>) -> (Vec<[i64; 2]>, bool, i64, Option<[Rb; 2]>) {
        let p = self.p.clone();
        self.stats.frames += 1;
        let expected = N_PREAMBLE + self.digital_nsym;
        let sym = match sym {
            Some(s) if s.len() == expected => s,
            _ => {
                self.stats.lost += 1;
                return (vec![[0, 0]; p.block], false, 2, None);
            }
        };
        let digital = &sym[N_PREAMBLE..N_PREAMBLE + self.digital_nsym];
        let (mut llr, mut hard, _var) = self.modem.demodulate(digital, self.lay.n_tx);
        for i in 0..llr.len() {
            if self.scr[i] == 1 {
                llr[i] = -llr[i];
            }
            hard[i] ^= self.scr[i];
        }
        let (bd, hdr) = match unpack_recovering(&self.lay, &llr, &hard) {
            Some(r) => r,
            None => {
                self.stats.crc_fail += 1;
                return (vec![[0, 0]; p.block], false, 2, None);
            }
        };
        if hdr.recovered > 0 {
            self.stats.recovered += 1;
        }
        let mut p1: Vec<i64> = self.lay.coef_stream_idx.iter().map(|&i| sigmoid_q15(llr[i])).collect();
        for (t, &ci) in self.lay.top_coef_idx.iter().enumerate() {
            p1[ci] = hdr.top_bits[t] as i64 * 32768;
        }
        let quality = if hdr.high_ok && !self.high_absent { 0 } else { 1 };
        if !hdr.high_ok {
            self.stats.high_crc_fail += 1;
        }
        let (coefs, rb) = self.soft_coeffs(&bd, &p1, hdr.high_ok);
        let m = haar_inverse(&coefs[0]);
        let s = haar_inverse(&coefs[1]);
        let out = (0..p.block).map(|i| ms_to_lr(m[i], s[i])).collect();
        (out, true, quality, Some(rb))
    }

    /// frames: 系統のフレームごとのシンボル（届かなければ None）。返り値: 時刻合わせ済みの音声（n）と有効フラグ。quality、rb_trace も更新する
    pub fn run(&mut self, frames: &[Option<Vec<[i64; 2]>>], n: usize, offset: usize) -> (Vec<[i64; 2]>, Vec<bool>) {
        let b = self.p.block;
        let h = b / 2;
        let nf = frames.len();
        let mut y = vec![[0i64; 2]; nf * b];
        let mut v = vec![false; nf * b];
        let mut rb = vec![[[INF_LOG; 5]; 2]; nf * b];
        self.quality = vec![0; nf];
        for (i, f) in frames.iter().enumerate() {
            let (blk, valid, q, last_rb) = self.decode(f.as_deref());
            y[i * b..(i + 1) * b].copy_from_slice(&blk);
            for t in 0..b {
                v[i * b + t] = valid;
            }
            self.quality[i] = q;
            if let Some(r) = last_rb {
                for t in 0..h {
                    rb[i * b + t] = r[0];
                    rb[i * b + h + t] = r[1];
                }
            }
        }
        let end = (offset + n).min(y.len());
        let start = offset.min(end);
        self.rb_trace = rb[start..end].to_vec();
        (y[start..end].to_vec(), v[start..end].to_vec())
    }
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

pub struct RxOutput {
    pub pcm: Vec<[i64; 2]>,
    pub quality: Vec<i64>,
    pub stats_a: Stats,
    pub stats_b: Option<Stats>,
    pub plc_samples: usize,
}

/// 2 系統（frames_b = None なら 1 系統）を受信して (n) の整数の音声を返す
pub fn receive(
    profile: &Profile, frames_a: &[Option<Vec<[i64; 2]>>], frames_b: Option<&[Option<Vec<[i64; 2]>>]>, n: usize, offset_b: usize,
    profile_b: Option<&Profile>,
) -> Result<RxOutput, String> {
    let pb = profile_b.unwrap_or(profile);
    if pb.fs != profile.fs || pb.block != profile.block {
        return Err("副系のサンプリング周波数とブロック長は主系と同じにしてください".into());
    }
    let mut da = StreamDecoder::new(profile)?;
    let (ya, va) = da.run(frames_a, n, 0);
    let mut db = None;
    let (y, v) = if let Some(fb) = frames_b {
        let mut d = StreamDecoder::new(pb)?;
        let (yb, vb) = d.run(fb, n, offset_b);
        let r = fuse_band(&ya, &va, &yb, &vb, profile.block, profile.levels, &da.rb_trace, &d.rb_trace);
        db = Some(d);
        r
    } else {
        (ya, va)
    };
    let plc_samples = v.iter().filter(|&&x| !x).count();
    let nblk = (y.len() + profile.block - 1) / profile.block;
    let quality = block_quality(profile.block, &da.quality, db.as_ref().map(|d| d.quality.as_slice()), offset_b, nblk);
    let mut traces: Vec<&Vec<Rb>> = vec![&da.rb_trace];
    if let Some(d) = &db {
        traces.push(&d.rb_trace);
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
    Ok(RxOutput { pcm, quality, stats_a: da.stats, stats_b: db.map(|d| d.stats), plc_samples })
}
