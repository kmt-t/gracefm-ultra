//! 浮動小数点の受信（Ultra の本番経路）。整数版（decoder.rs、receiver.rs）と同じ鏡の構造で、演算だけを浮動小数点（f64）にしたもの。
//! 違い: LLR とビタビは浮動小数点、p1 は厳密な sigmoid、逆 Haar は直交（1/√2）、誤差分散は線形の値、融合は逆分散重み、
//! 欠落補間（LPC）は倍精度、出力 IIR は真の Butterworth（4 次、4 kHz）。信号の尺度は満点 ±1。

use crate::codec::quant::{dequantize_soft_f, RbF};
use crate::codec::{haar_forward_f, haar_inverse_f, synthesize_f};
use crate::decoder::{Stats, Q_LOST, Q_NO_HIGH, Q_OK};
use crate::framing::{unpack_f, unpack_recovering_with, Unpacked};
use crate::layout::Layout;
use crate::modem::{Modem, N_PREAMBLE, SYM_FRAC};
use crate::profiles::Profile;
use crate::scrambler::Scrambler;

/// 整数のシンボル（Q22）を浮動小数点（±1）にする
pub fn sym_to_f(sym: &[[i64; 2]]) -> Vec<[f64; 2]> {
    let one = (1u64 << SYM_FRAC) as f64;
    sym.iter().map(|s| [s[0] as f64 / one, s[1] as f64 / one]).collect()
}

pub struct FFrame {
    pub pcm: Vec<[f64; 2]>,
    pub valid: bool,
    pub quality: i64,
    pub rb: Option<[RbF; 2]>,
}

pub struct FStream {
    pub pcm: Vec<[f64; 2]>,
    pub valid: Vec<bool>,
    pub quality: Vec<i64>,
    /// サンプルごとの誤差分散（線形）[成分][帯域]。届かなければ INFINITY
    pub rb_trace: Vec<RbF>,
}

pub struct FDecoder {
    pub p: Profile,
    pub lay: Layout,
    modem: Modem,
    scrambler: Scrambler,
    digital_nsym: usize,
    high_absent: bool,
    pub stats: Stats,
}

impl FDecoder {
    pub fn new(p: &Profile) -> Result<FDecoder, String> {
        let lay = Layout::new(p)?;
        let modem = Modem::new(p.qam16h, p.alpha);
        let digital_nsym = modem.n_symbols(lay.n_tx, lay.n_strong);
        let scrambler = Scrambler::new(digital_nsym * modem.bits_per_symbol());
        let high_absent = p.bits_m[p.high_k()..].iter().all(|&b| b == 0) && p.bits_s[p.high_k()..].iter().all(|&b| b == 0);
        Ok(FDecoder { p: p.clone(), lay, modem, scrambler, digital_nsym, high_absent, stats: Stats::default() })
    }

    fn lost(&self) -> FFrame {
        FFrame { pcm: vec![[0.0, 0.0]; self.p.block], valid: false, quality: Q_LOST, rb: None }
    }

    /// 1 フレームの受信シンボル（プリアンブル込み、±1 = 1.0。届かなければ None）→ 復号結果
    pub fn decode_frame(&mut self, sym: Option<&[[f64; 2]]>) -> FFrame {
        self.stats.frames += 1;
        let expected = N_PREAMBLE + self.digital_nsym;
        let sym = match sym {
            Some(s) if s.len() == expected => s,
            _ => {
                self.stats.lost += 1;
                return self.lost();
            }
        };
        let (mut llr, mut hard, _var) = self.modem.demodulate_f(&sym[N_PREAMBLE..expected], self.lay.n_tx);
        self.scrambler.descramble_f(&mut llr, &mut hard);
        let lay = &self.lay;
        let r: Option<(Unpacked, usize)> =
            unpack_recovering_with(lay, &llr, &hard, &|v: f64| -v, &|v: f64| v.abs(), &|s, b| unpack_f(lay, s, b));
        let (u, recovered) = match r {
            Some(r) => r,
            None => {
                self.stats.crc_fail += 1;
                return self.lost();
            }
        };
        if recovered > 0 {
            self.stats.recovered += 1;
        }
        // 1 である確率 = 1 / (1 + e^LLR)（LLR が正なら 0 らしい）
        let mut p1: Vec<f64> = lay.coef_stream_idx.iter().map(|&i| 1.0 / (1.0 + llr[i].clamp(-40.0, 40.0).exp())).collect();
        for (t, &ci) in lay.top_coef_idx.iter().enumerate() {
            p1[ci] = u.protected_bits[t] as f64; // FEC で訂正した保護ビットは確定値
        }
        let quality = if u.high_ok && !self.high_absent { Q_OK } else { Q_NO_HIGH };
        if !u.high_ok {
            self.stats.high_crc_fail += 1;
        }
        let est = dequantize_soft_f(&u.bd, lay, &p1, u.high_ok);
        FFrame { pcm: synthesize_f(&est.coef), valid: true, quality, rb: Some(est.rb) }
    }

    pub fn run(&mut self, frames: &[Option<Vec<[f64; 2]>>], n: usize, offset: usize) -> FStream {
        let b = self.p.block;
        let h = b / 2;
        let nf = frames.len();
        let mut pcm = vec![[0f64; 2]; nf * b];
        let mut valid = vec![false; nf * b];
        let mut rb = vec![[[f64::INFINITY; 5]; 2]; nf * b];
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
        FStream { pcm: pcm[start..end].to_vec(), valid: valid[start..end].to_vec(), quality, rb_trace: rb[start..end].to_vec() }
    }
}

// ---------------------------------------------------------------- 2 系統の融合（逆分散重み）

/// w_A = R_B / (R_A + R_B)。情報なし（INFINITY）は重み 0 / 1 / 1/2
pub fn weight_a_f(ra: f64, rb: f64) -> f64 {
    match (ra.is_infinite(), rb.is_infinite()) {
        (true, true) => 0.5,
        (true, false) => 0.0,
        (false, true) => 1.0,
        _ => rb / (ra + rb),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fuse_band_f(
    ya: &[[f64; 2]], va: &[bool], yb: &[[f64; 2]], vb: &[bool], seg: usize, levels: usize, rba: &[RbF], rbb: &[RbF],
) -> (Vec<[f64; 2]>, Vec<bool>) {
    let n = ya.len();
    let nb = levels + 1;
    let ns = (n + seg - 1) / seg;
    let h = seg / 2;
    let ms = |y: &[[f64; 2]], blk: usize, ci: usize, t: usize| -> f64 {
        let i = blk * seg + t;
        let (l, r) = if i < y.len() { (y[i][0], y[i][1]) } else { (0.0, 0.0) };
        if ci == 0 { (l + r) * 0.5 } else { (l - r) * 0.5 }
    };
    let blk_valid = |v: &[bool], blk: usize| -> bool { blk * seg < v.len() && v[blk * seg] };
    let rb_at = |rb: &[RbF], blk: usize, off: usize, ci: usize, k: usize| -> f64 {
        let i = blk * seg + off;
        if i < rb.len() { rb[i][ci][k] } else { f64::INFINITY }
    };
    let mut out_ms = vec![[vec![0f64; seg], vec![0f64; seg]]; ns];
    for blk in 0..ns {
        let (vab, vbb) = (blk_valid(va, blk), blk_valid(vb, blk));
        let (only_a, only_b) = (vab && !vbb, vbb && !vab);
        let ov = |w: f64| -> f64 { if only_a { 1.0 } else if only_b { 0.0 } else { w } };
        for ci in 0..2 {
            let xa: Vec<f64> = (0..seg).map(|t| ms(ya, blk, ci, t)).collect();
            let xb: Vec<f64> = (0..seg).map(|t| ms(yb, blk, ci, t)).collect();
            let ba = haar_forward_f(&xa, levels);
            let bb = haar_forward_f(&xb, levels);
            let mut bands = Vec::with_capacity(nb);
            for k in 0..nb {
                let nn = ba[k].len();
                let w1 = weight_a_f(rb_at(rba, blk, 0, ci, k), rb_at(rbb, blk, 0, ci, k));
                let w2 = weight_a_f(rb_at(rba, blk, h, ci, k), rb_at(rbb, blk, h, ci, k));
                let band: Vec<f64> = (0..nn)
                    .map(|j| {
                        let w = if nn == 1 { ov((w1 + w2) * 0.5) } else if j < nn / 2 { ov(w1) } else { ov(w2) };
                        w * ba[k][j] + (1.0 - w) * bb[k][j]
                    })
                    .collect();
                bands.push(band);
            }
            out_ms[blk][ci] = haar_inverse_f(&bands);
        }
    }
    let mut y = Vec::with_capacity(n);
    let mut valid = Vec::with_capacity(n);
    for i in 0..n {
        let blk = i / seg;
        let v = blk_valid(va, blk) || blk_valid(vb, blk);
        valid.push(v);
        let (m, s) = (out_ms[blk][0][i % seg], out_ms[blk][1][i % seg]);
        y.push(if v { [m + s, m - s] } else { [0.0, 0.0] });
    }
    (y, valid)
}

// ---------------------------------------------------------------- 欠落補間（LPC 外挿、倍精度）

pub struct PlcParamsF {
    pub order: usize,
    pub hist_len: usize,
    pub hold_ms: f64,
    pub fade_ms: f64,
    pub xfade: usize,
}

impl Default for PlcParamsF {
    fn default() -> Self {
        PlcParamsF { order: 16, hist_len: 128, hold_ms: 10.0, fade_ms: 20.0, xfade: 16 }
    }
}

/// ハン窓をかけた履歴の自己相関から Levinson-Durbin で LPC 係数 a[0..=order]（a[0] = 1）を求める。白色雑音補正と帯域拡張（0.998^i）つき
pub fn lpc_f(x: &[f64], order: usize) -> Vec<f64> {
    let n = x.len();
    let denom = (n.max(2) - 1) as f64;
    let w: Vec<f64> = x.iter().enumerate().map(|(i, &v)| v * 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos())).collect();
    let mut r: Vec<f64> = (0..=order).map(|k| (0..n.saturating_sub(k)).map(|i| w[i] * w[i + k]).sum()).collect();
    let mut a = vec![0f64; order + 1];
    a[0] = 1.0;
    if r[0] <= 0.0 {
        return a;
    }
    r[0] *= 1.0 + 1e-4;
    let mut err = r[0];
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = (-acc / err).clamp(-0.99997, 0.99997);
        let old = a.clone();
        for j in 1..i {
            a[j] = old[j] + k * old[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            break;
        }
    }
    let mut bw = 1.0;
    for i in 1..=order {
        bw *= 0.998;
        a[i] *= bw;
    }
    a
}

pub fn extrapolate_f(hist: &[f64], n: usize, order: usize) -> Vec<f64> {
    if hist.len() <= order {
        return vec![0.0; n];
    }
    let a = lpc_f(hist, order);
    let mut buf: Vec<f64> = hist[hist.len() - order..].to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut acc = 0.0;
        for j in 0..order {
            acc += a[1 + j] * buf[buf.len() - 1 - j];
        }
        let v = (-acc).clamp(-1.0, 1.0);
        out.push(v);
        buf.push(v);
        buf.remove(0);
    }
    out
}

/// 欠落区間を LPC 外挿で埋める。hold_ms までは等倍、その後 fade_ms で 0 へ。復帰時は xfade サンプルで外挿値から受信値へクロスフェード
pub fn conceal_f(y: &[[f64; 2]], valid: &[bool], fs: usize, pp: &PlcParamsF) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = y.iter().zip(valid).map(|(s, &v)| if v { *s } else { [0.0, 0.0] }).collect();
    let hold = fs as f64 * pp.hold_ms / 1000.0;
    let fade = (fs as f64 * pp.fade_ms / 1000.0).max(1.0);
    let n = y.len();
    let mut i = 0;
    while i < n {
        if valid[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !valid[j] {
            j += 1;
        }
        let l = j - i;
        let k = pp.xfade.min(n - j);
        for c in 0..2 {
            let hist: Vec<f64> = out[i.saturating_sub(pp.hist_len)..i].iter().map(|v| v[c]).collect();
            let ext = extrapolate_f(&hist, l + k, pp.order);
            let pred: Vec<f64> = ext.iter().enumerate().map(|(t, &e)| e * (1.0 - (t as f64 - hold) / fade).clamp(0.0, 1.0)).collect();
            for t in 0..l {
                out[i + t][c] = pred[t];
            }
            for m in 0..k {
                let w = (m as f64 + 1.0) / (k as f64 + 1.0);
                out[j + m][c] = w * out[j + m][c] + (1.0 - w) * pred[l + m];
            }
        }
        i = j;
    }
    out
}

pub fn crossfade_switch_f(y: &[[f64; 2]], patterns: &[Vec<bool>], block: usize, pp: &PlcParamsF) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = y.to_vec();
    let nb = patterns.len().min(y.len() / block);
    for i in 1..nb {
        if patterns[i] != patterns[i - 1] && patterns[i].iter().any(|&b| b) && patterns[i - 1].iter().any(|&b| b) {
            let s = i * block;
            let k = pp.xfade.min(y.len() - s);
            for c in 0..2 {
                let hist: Vec<f64> = out[s.saturating_sub(pp.hist_len)..s].iter().map(|v| v[c]).collect();
                let pred = extrapolate_f(&hist, k, pp.order);
                for m in 0..k {
                    let w = (m as f64 + 1.0) / (k as f64 + 1.0);
                    out[s + m][c] = w * out[s + m][c] + (1.0 - w) * pred[m];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- 出力 IIR（真の Butterworth）

/// 4 次 Butterworth ローパス（カットオフ fc [Hz]）の 2 つの biquad（正規化済み、a0 = 1）: [b0, b1, b2, a1, a2]
pub fn butter4_lowpass(fc: f64, fs: f64) -> [[f64; 5]; 2] {
    let w0 = 2.0 * std::f64::consts::PI * fc / fs;
    let (cw, sw) = (w0.cos(), w0.sin());
    let q = [0.5 / (std::f64::consts::PI / 8.0).cos(), 0.5 / (3.0 * std::f64::consts::PI / 8.0).cos()];
    let mut out = [[0.0; 5]; 2];
    for (i, &qi) in q.iter().enumerate() {
        let alpha = sw / (2.0 * qi);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cw) / 2.0 / a0;
        out[i] = [b0, 2.0 * b0, b0, -2.0 * cw / a0, (1.0 - alpha) / a0];
    }
    out
}

/// 直接形 II の転置で biquad を直列に通す（因果）
pub fn iir_filter_f(x: &[[f64; 2]], sections: &[[f64; 5]]) -> Vec<[f64; 2]> {
    let mut st = vec![[[0f64; 2]; 2]; sections.len()];
    let mut out = Vec::with_capacity(x.len());
    for xs in x {
        let mut o = [0f64; 2];
        for c in 0..2 {
            let mut v = xs[c];
            for (j, sec) in sections.iter().enumerate() {
                let s = &mut st[j][c];
                let y = sec[0] * v + s[0];
                s[0] = sec[1] * v - sec[3] * y + s[1];
                s[1] = sec[2] * v - sec[4] * y;
                v = y;
            }
            o[c] = v;
        }
        out.push(o);
    }
    out
}

/// 品質コード（0 正常, 1 高域欠落, 2 欠落）に応じて y と alt を切り替える。切り替えは、そのブロックの先頭から fade サンプルの直線
pub fn select_by_quality_f(y: &[[f64; 2]], alt: &[[f64; 2]], quality: &[i64], block: usize, fade: usize) -> Vec<[f64; 2]> {
    let n = y.len();
    let mut out = y.to_vec();
    let mut w = 0.0f64;
    for (i, &q) in quality.iter().enumerate() {
        let s0 = i * block;
        if s0 >= n {
            break;
        }
        let s1 = n.min(s0 + block);
        let target = if q == 1 || q == 2 { 1.0 } else { 0.0 };
        let mut r = w;
        for t in 0..s1 - s0 {
            let m = ((t + 1) as f64).min(fade as f64);
            r = w + (target - w) * m / fade as f64;
            for c in 0..2 {
                out[s0 + t][c] = y[s0 + t][c] * (1.0 - r) + alt[s0 + t][c] * r;
            }
        }
        w = r;
    }
    out
}

/// 欠落・高域欠落のブロックだけ IIR の出力へ切り替える。出力は全体を alt_delay サンプル遅らせて揃える（整数版と同じ）
pub fn apply_alt_f(y: &[[f64; 2]], quality: &[i64], fs: usize, block: usize) -> Vec<[f64; 2]> {
    let d = crate::iir::alt_delay(fs);
    let alt = iir_filter_f(y, &butter4_lowpass(4000.0, fs as f64));
    let mut delayed = vec![[0f64; 2]; d];
    delayed.extend_from_slice(&y[..y.len() - d]);
    select_by_quality_f(&delayed, &alt, quality, block, fs / 1000)
}

// ---------------------------------------------------------------- 出力段

pub struct FRxOutput {
    pub pcm: Vec<[f64; 2]>,
    pub quality: Vec<i64>,
    pub stats_a: Stats,
    pub stats_b: Option<Stats>,
    pub plc_samples: usize,
}

/// 2 系統（frames_b = None なら 1 系統）を受信して n サンプルの音声（満点 ±1）を返す
pub fn receive_f(
    profile: &Profile, frames_a: &[Option<Vec<[f64; 2]>>], frames_b: Option<&[Option<Vec<[f64; 2]>>]>, n: usize, offset_b: usize,
    profile_b: Option<&Profile>,
) -> Result<FRxOutput, String> {
    let pb = profile_b.unwrap_or(profile);
    if pb.fs != profile.fs || pb.block != profile.block {
        return Err("副系のサンプリング周波数とブロック長は主系と同じにしてください".into());
    }
    let mut da = FDecoder::new(profile)?;
    let a = da.run(frames_a, n, 0);
    let mut db: Option<(FDecoder, FStream)> = None;
    let (y, v) = if let Some(fb) = frames_b {
        let mut d = FDecoder::new(pb)?;
        let b = d.run(fb, n, offset_b);
        let r = fuse_band_f(&a.pcm, &a.valid, &b.pcm, &b.valid, profile.block, profile.levels, &a.rb_trace, &b.rb_trace);
        db = Some((d, b));
        r
    } else {
        (a.pcm.clone(), a.valid.clone())
    };
    let plc_samples = v.iter().filter(|&&x| !x).count();
    let nblk = (y.len() + profile.block - 1) / profile.block;
    let quality = crate::receiver::block_quality(profile.block, &a.quality, db.as_ref().map(|(_, b)| b.quality.as_slice()), offset_b, nblk);
    let mut traces = vec![&a.rb_trace];
    if let Some((_, b)) = &db {
        traces.push(&b.rb_trace);
    }
    let patterns: Vec<Vec<bool>> = (0..y.len() / profile.block)
        .map(|i| {
            let mut pat = Vec::new();
            for rb in &traces {
                let r = rb.get(i * profile.block).copied().unwrap_or([[f64::INFINITY; 5]; 2]);
                for ci in 0..2 {
                    for k in 0..profile.nbands() {
                        pat.push(r[ci][k].is_finite());
                    }
                }
            }
            pat
        })
        .collect();
    let pp = PlcParamsF::default();
    let y = crossfade_switch_f(&y, &patterns, profile.block, &pp);
    let y = conceal_f(&y, &v, profile.fs as usize, &pp);
    let pcm = apply_alt_f(&y, &quality, profile.fs as usize, profile.block);
    Ok(FRxOutput { pcm, quality, stats_a: da.stats, stats_b: db.map(|(d, _)| d.stats), plc_samples })
}
