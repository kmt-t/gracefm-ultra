//! 受信: フレームのシンボル → 帯域係数と、その確からしさ（浮動小数点）。GraceFM `gracefm/decoder.py` に相当（暗号化は未対応）。
//! decode_frame = demodulate → descramble → unpack（ヘッダは事前知識つき、通らなければリスト復号）→ soft_coeffs（軟判定の逆量子化）。エンコーダ（encoder.rs）の鏡。
//! 音声への合成と、欠落・誤りの補正は下流（receiver.rs）。信号の尺度は満点 ±1。

use crate::codec::exp_range;
use crate::codec::quant::{offset_of, qmax_of, qmin_of};
use crate::codec::BlockData;
use crate::fec;
use crate::framing::{unpack, UnpackOptions};
use crate::layout::Layout;
use crate::modem::{prbs, y_estimate, y_level, y_level_llr, y_phase_llr, Modem, N_PREAMBLE};
use crate::prior::Tracker;
use crate::profiles::{Profile, MAX_BANDS};
use crate::scrambler::Scrambler;

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

/// 帯域ごとの誤差分散（線形、満点 1.0 に対する二乗）: [成分][帯域]。情報なしは f64::INFINITY
pub type BandVar = [[f64; MAX_BANDS]; 2];

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub frames: usize,
    pub lost: usize,
    pub crc_fail: usize,
    pub high_crc_fail: usize,
    /// ヘッダを、最尤でない候補（リスト復号）で採ったフレームの数
    pub recovered: usize,
}

/// 基本 CRC が通らないフレームの、尺度なしの手がかり。ヘッダ（指数）が分からなくても、仮数の受信 LLR から、帯域内の係数の相対的な形が残る
#[derive(Clone, Debug)]
pub struct Hint {
    /// 係数ごとの仮数の期待値 / qmax（[-1, 1] 付近）。[成分][帯域を並べた 64 係数]。送らない帯域は 0
    pub h: [Vec<f64>; 2],
    /// 帯域ごとの、仮数の事後標準偏差の平均 / qmax（0〜1 に収める）。送らない帯域は 0
    pub q: [[f64; MAX_BANDS]; 2],
}

pub struct DecodedFrame {
    pub status: Status,
    /// 帯域係数 [成分][帯域][j]（満点 1.0）。基本 CRC が通らない・届かないフレームは 0、高域欠落のフレームは高域が 0
    pub coef: Vec<Vec<Vec<f64>>>,
    /// 誤差分散。情報なし（届かない・基本 CRC 不一致・高域欠落の高域）は INFINITY
    pub var: BandVar,
    /// 観測のビットの受信 LLR（自然対数、正 = 0 らしい。スクランブル解除後、FEC 前）。並びは Layout の obs_pos。届かないフレームは空
    pub llr: Vec<f64>,
    /// ヘッダのフレーム番号と、BlockData（基本 CRC が通ったとき）
    pub header: Option<(u64, BlockData)>,
    /// 採用したヘッダの、最尤との対数尤度の差 [nat]（`Unpacked::gap`）。届かない・CRC 不一致は 0
    pub gap: f64,
    /// 基本 CRC が通らないフレームの手がかり
    pub hint: Option<Hint>,
}

/// BlockData（指数とアテネーション）から、係数ごとの量子化の刻み（満点 = 2^22 の尺度）を out（成分ごとに全帯域を並べた 2 × ブロック長の並び）に書く。
/// 指数やアテネーションがない帯域は 0 のまま。high_ok が false のときは、高域（high_k 以上の帯域）のヘッダは信用できないので 0 のまま
pub fn block_steps(lay: &Layout, bd: &BlockData, high_ok: bool, out: &mut [f32]) {
    let p = &lay.p;
    for &(ch, k) in &lay.coef_groups {
        if k >= p.high_k() && !high_ok {
            continue;
        }
        let (Some(e), Some(attn)) = (bd.eidx[ch][k], bd.attn[ch][k].as_ref()) else { continue };
        let e = e + exp_range(p.bits(ch)[k], p).0;
        let o = ch * p.block + (0..k).map(|i| p.ncoef(i)).sum::<usize>();
        for j in 0..p.ncoef(k) {
            out[o + j] = (2f64.powi((e - p.attn_at(ch, k, attn, j)) as i32) * (1u64 << 22) as f64) as f32;
        }
    }
}

/// 軟判定の逆量子化。p1: 仮数の各ビットが 1 である確率（coef_items の並び）。
/// 期待値は 仮数 + Σ (p1 − 硬判定ビット)·2^pos、誤差分散は (Σ p1(1−p1)·4^pos + 1/12)·刻み²。
/// y: Y 帯域の係数ごとの（半径の段の推定値、その誤差分散）。high_ok が false のときは、高域（high_k 以上の帯域）を 0（情報なし）にする。返り値: (帯域係数 [ch][k][j]、帯域ごとの誤差分散（係数の平均）)
pub fn soft_coeffs(lay: &Layout, bd: &BlockData, p1: &[f64], y: &[(f64, f64)], high_ok: bool) -> (Vec<Vec<Vec<f64>>>, BandVar) {
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
    for (&cf, &(d, v)) in lay.y_flat.iter().zip(y) {
        du[cf] = d;
        var[cf] = v;
    }
    let mut coef = vec![vec![Vec::new(); nb]; 2];
    let mut var_b = [[f64::INFINITY; MAX_BANDS]; 2];
    for ch in 0..2 {
        for k in 0..nb {
            let b = p.bits(ch)[k];
            let n = p.ncoef(k);
            if b == 0 || (k >= p.high_k() && !high_ok) {
                coef[ch][k] = vec![0.0; n];
                continue;
            }
            let o = lay.coef_offset[&(ch, k)];
            let (qmax, qmin) = (qmax_of(b) as f64, qmin_of(b, p, ch, k) as f64);
            let mant = bd.mant[ch][k].as_ref().unwrap();
            let attn = bd.attn[ch][k].as_ref().unwrap();
            let e = bd.eidx[ch][k].unwrap() + exp_range(b, p).0; // 刻みの指数（満点 = 1.0）
            let mut c = Vec::with_capacity(n);
            let mut vs = 0.0;
            for j in 0..n {
                let m = (mant[j] as f64 + du[o + j]).clamp(qmin, qmax);
                let a = p.attn_at(ch, k, attn, j);
                let step = 2f64.powi((e - a) as i32);
                c.push(m * step);
                vs += (var[o + j] + 1.0 / 12.0) * step * step;
            }
            var_b[ch][k] = vs / n as f64;
            coef[ch][k] = c;
        }
    }
    (coef, var_b)
}

/// 1 系統の復号。フレームを順に push する（ヘッダの事前知識のために、直前までのヘッダを覚える）
pub struct Decoder {
    pub p: Profile,
    pub lay: Layout,
    modem: Modem,
    scrambler: Scrambler,
    digital_nsym: usize,
    y_mask: Vec<u8>,
    tracker: Tracker,
    pub stats: Stats,
}

impl Decoder {
    pub fn new(p: &Profile) -> Result<Decoder, String> {
        let lay = Layout::new(p)?;
        let modem = Modem::new(p.qam16h, p.alpha);
        let digital_nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
        let digital_bits = digital_nsym * modem.bits_per_symbol();
        let scrambler = Scrambler::new(digital_bits);
        let y_mask = prbs(digital_bits + 3 * lay.y_items.len()).split_off(digital_bits);
        Ok(Decoder { p: p.clone(), lay, modem, scrambler, digital_nsym, y_mask, tracker: Tracker::new(), stats: Stats::default() })
    }

    fn missing(&self, status: Status, llr: Vec<f64>) -> DecodedFrame {
        let coef = (0..2).map(|_| (0..self.p.nbands()).map(|k| vec![0.0; self.p.ncoef(k)]).collect()).collect();
        DecodedFrame { status, coef, var: [[f64::INFINITY; MAX_BANDS]; 2], llr, header: None, gap: 0.0, hint: None }
    }

    /// 観測のビットの受信 LLR（obs_pos の並び）から、尺度なしの手がかりを作る
    fn hint(&self, llr: &[f64]) -> Hint {
        let lay = &self.lay;
        let p = &self.p;
        let mut eu = vec![0f64; lay.n_coef];
        let mut var = vec![0f64; lay.n_coef];
        for (j, &cf) in lay.obs_flat.iter().enumerate() {
            let p1 = 1.0 / (1.0 + llr[j].clamp(-40.0, 40.0).exp());
            let w = (1u64 << lay.obs_pos[j]) as f64;
            eu[cf] += p1 * w;
            var[cf] += p1 * (1.0 - p1) * w * w;
        }
        let mut hint = Hint { h: [vec![0.0; 64], vec![0.0; 64]], q: [[0.0; MAX_BANDS]; 2] };
        for ch in 0..2 {
            let mut base = 0;
            for k in 0..p.nbands() {
                let n = p.ncoef(k);
                let b = p.bits(ch)[k];
                if b > 0 {
                    let (o, qmax, off) = (lay.coef_offset[&(ch, k)], qmax_of(b) as f64, offset_of(b, p, ch, k) as f64);
                    for j in 0..n {
                        hint.h[ch][base + j] = (eu[o + j] - off) / qmax;
                        hint.q[ch][k] += var[o + j].sqrt() / qmax;
                    }
                    hint.q[ch][k] = (hint.q[ch][k] / n as f64).clamp(0.0, 1.0);
                }
                base += n;
            }
        }
        hint
    }

    /// 1 フレームの受信シンボル（プリアンブル込み、±1 = 1.0。届かなければ None）→ 復号結果。返り値の 2 つ目は、採用したヘッダの候補の順位（最尤が CRC を通らなかったとき > 0）。
    /// opt: ヘッダの事前の軟値（`Tracker::header_prior`）と、リスト復号の設定
    pub fn decode_frame(&self, sym: Option<&[[f64; 2]]>, opt: &UnpackOptions) -> (DecodedFrame, usize) {
        let y_start = N_PREAMBLE + self.digital_nsym;
        let expected = y_start + self.lay.y_items.len();
        let sym = match sym {
            Some(s) if s.len() == expected => s,
            _ => return (self.missing(Status::Lost, Vec::new()), 0),
        };
        let (mut llr, mut hard, var) = self.modem.demodulate(&sym[N_PREAMBLE..y_start], self.lay.n_tx);
        let y_sym = &sym[y_start..];
        let (y_msb_llr, y_par_llr) = y_phase_llr(y_sym, &self.y_mask, var);
        self.scrambler.descramble(&mut llr, &mut hard);
        let lay = &self.lay;
        let mut mant: Vec<f64> = lay.coef_stream_idx.iter().map(|&i| llr[i]).collect();
        let hard_bit = |l: &f64| (*l < 0.0) as u8;
        let opt = UnpackOptions { y_soft: Some((&y_msb_llr, &y_par_llr)), ..*opt };
        let u = match unpack(lay, &llr, &hard, &opt) {
            Some(r) => r,
            None => {
                let (msb, par): (Vec<u8>, Vec<u8>) = (y_msb_llr.iter().map(hard_bit).collect(), y_par_llr.iter().map(hard_bit).collect());
                self.push_y_llr(&mut mant, y_sym, &msb, &par, &y_msb_llr, var);
                let hint = self.hint(&mant);
                let mut d = self.missing(Status::CrcFail, mant);
                d.hint = Some(hint);
                return (d, 0);
            }
        };
        // 1 である確率 = 1 / (1 + e^LLR)（LLR が正なら 0 らしい）
        let mut p1: Vec<f64> = mant.iter().map(|&l| 1.0 / (1.0 + l.clamp(-40.0, 40.0).exp())).collect();
        for (t, &ci) in lay.top_coef_idx.iter().enumerate() {
            p1[ci] = u.top_bits[t] as f64; // FEC で訂正した保護ビットは確定値
        }
        let status = if u.high_ok { Status::Ok } else { Status::HighFail };
        let parity = if u.y_msb.is_empty() { Vec::new() } else { fec::rsc_parity(&u.y_msb).0[..u.y_msb.len()].to_vec() };
        self.push_y_llr(&mut mant, y_sym, &u.y_msb, &parity, &y_msb_llr, var);
        let y: Vec<(f64, f64)> = y_sym
            .iter()
            .enumerate()
            .map(|(t, s)| {
                let (obs, noise) = y_level(s, u.y_msb[t], parity[t], &self.y_mask[3 * t..3 * t + 3], var);
                y_estimate(obs, noise)
            })
            .collect();
        let (coef, var) = soft_coeffs(lay, &u.bd, &p1, &y, u.high_ok);
        (DecodedFrame { status, coef, var, llr: mant, header: Some((u.counter, u.bd)), gap: u.gap, hint: None }, u.rank)
    }

    /// 観測の LLR に、Y の係数ごとの全ビットを足す: 最上位ビットは位相、残りは半径の段から
    fn push_y_llr(&self, out: &mut Vec<f64>, sym: &[[f64; 2]], msb: &[u8], parity: &[u8], msb_llr: &[f64], var: f64) {
        for (t, s) in sym.iter().enumerate() {
            let (obs, noise) = y_level(s, msb[t], parity[t], &self.y_mask[3 * t..3 * t + 3], var);
            out.push(msb_llr[t]);
            out.extend(y_level_llr(obs, noise));
        }
    }

    /// 次のフレームを復号する。ヘッダの事前知識は、直前までに復号したヘッダから作る
    pub fn push(&mut self, sym: Option<&[[f64; 2]]>) -> DecodedFrame {
        let i = self.stats.frames;
        let prior = self.tracker.header_prior(&self.lay, i);
        let (d, recovered) = self.decode_frame(sym, &UnpackOptions { header_prior: Some(&prior), ..UnpackOptions::default() });
        if let Some((counter, bd)) = &d.header {
            self.tracker.observe(i, *counter, bd);
        }
        let st = &mut self.stats;
        st.frames += 1;
        match d.status {
            Status::Lost => st.lost += 1,
            Status::CrcFail => st.crc_fail += 1,
            Status::HighFail => st.high_crc_fail += 1,
            Status::Ok => {}
        }
        st.recovered += (recovered > 0) as usize;
        d
    }
}
