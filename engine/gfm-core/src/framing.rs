//! フレームの送信（GraceFM `gracefm/framing.py` の pack の移植。暗号化なし）。

use crate::codec::{offset_of, BlockData};
use crate::fec;
use crate::layout::Layout;

/// 仮数の各ビット（Layout の coef_items の並び）。pack が使う
pub fn coefficient_bits(lay: &Layout, bd: &BlockData) -> Vec<u8> {
    let mut flat_mant: Vec<i64> = Vec::with_capacity(lay.n_coef);
    for &(ch, k) in &lay.coef_groups {
        flat_mant.extend(bd.mant[ch][k].as_ref().unwrap());
    }
    (0..lay.n_coef_bits()).map(|i| (((flat_mant[lay.coef_flat[i]] + lay.coef_offset_bits[i]) >> lay.coef_pos[i]) & 1) as u8).collect()
}

/// Y 帯域の仮数（オフセットバイナリ。最上位ビットが位相、残りが半径の段）
pub fn y_u(lay: &Layout, bd: &BlockData) -> Vec<i64> {
    let p = &lay.p;
    lay.y_items.iter().map(|&(ch, k, j)| bd.mant[ch][k].as_ref().unwrap()[j] + offset_of(p.bits(ch)[k], p, ch, k)).collect()
}

/// Y 帯域の最上位ビット
pub fn y_msb(lay: &Layout, u: &[i64]) -> Vec<u8> {
    u.iter().enumerate().map(|(t, &v)| (v >> (lay.p.bits(lay.y_items[t].0)[lay.y_items[t].1] - 1)) as u8).collect()
}

/// 受信の観測の並び（Layout の obs_pos）の、送ったビット
pub fn observed_bits(lay: &Layout, bd: &BlockData) -> Vec<u8> {
    let mut out = coefficient_bits(lay, bd);
    let u = y_u(lay, bd);
    let mut t = 0;
    for pos in &lay.obs_pos[lay.n_coef_bits()..] {
        out.push(((u[t] >> pos) & 1) as u8);
        t += (*pos == 0) as usize;
    }
    out
}

/// BlockData → 送信ビット列（n_tx 個、スクランブル前）
pub fn pack(lay: &Layout, bd: &BlockData, stream_id: u64, counter: u64) -> Vec<u8> {
    let mut prot = lay.header_prefix(stream_id, counter);
    for &(ch, k) in &lay.exp_items {
        prot.extend(lay.exp_bits_of(bd.eidx[ch][k].unwrap()));
    }
    for &(ch, k, j) in &lay.attn_items {
        prot.push(bd.attn[ch][k].as_ref().unwrap()[j] as u8);
    }
    let coef = coefficient_bits(lay, bd);
    let top: Vec<u8> = lay.top_coef_idx.iter().map(|&i| coef[i]).collect();
    let ymsb = y_msb(lay, &y_u(lay, bd));
    let mut core: Vec<u8> = lay.core_prot.iter().map(|&i| prot[i]).collect();
    core.extend(lay.core_top.iter().map(|&i| top[i]));
    core.extend(lay.core_y.iter().map(|&i| ymsb[i]));
    let mut high: Vec<u8> = lay.high_prot.iter().map(|&i| prot[i]).collect();
    high.extend(lay.high_top.iter().map(|&i| top[i]));
    high.extend(lay.high_y.iter().map(|&i| ymsb[i]));
    let mut header_x = prot;
    header_x.extend(fec::crc_core(&core));
    if lay.has_high_crc {
        header_x.extend(fec::crc_high(&high));
    }
    let mut stream = vec![0u8; lay.n_tx];
    let (hp, ht) = fec::rsc_parity(&header_x);
    for (i, &v) in hp.iter().enumerate() {
        stream[lay.hpar_idx[i]] = v;
    }
    for (i, &v) in header_x.iter().enumerate() {
        stream[lay.hdata_idx[i]] = v;
    }
    for (i, &v) in ht.iter().enumerate() {
        stream[lay.htail_idx[i]] = v;
    }
    if lay.n_top > 0 {
        let (par, tail) = fec::rsc_parity(&top);
        for i in 0..lay.n_top {
            stream[lay.parity_idx[i]] = par[i];
        }
        let tail_all: Vec<u8> = par[lay.n_top..].iter().chain(tail.iter()).copied().collect();
        for (i, &v) in tail_all.iter().enumerate() {
            stream[lay.tail_idx[i]] = v;
        }
    }
    if !ymsb.is_empty() {
        let (par, tail) = fec::rsc_parity(&ymsb);
        let tail_all: Vec<u8> = par[ymsb.len()..].iter().chain(tail.iter()).copied().collect();
        for (i, &v) in tail_all.iter().enumerate() {
            stream[lay.ytail_idx[i]] = v;
        }
    }
    for (i, &v) in coef.iter().enumerate() {
        stream[lay.coef_stream_idx[i]] = v;
    }
    stream
}

// ---------------------------------------------------------------- 受信

/// unpack の出力。top_bits は FEC で訂正した保護ビット（上位 K bit）で、受信側が事後確率を確定値にするのに使う
pub struct Unpacked {
    pub bd: BlockData,
    pub high_ok: bool,
    pub top_bits: Vec<u8>,
    /// Y 帯域の最上位ビット（訂正後）
    pub y_msb: Vec<u8>,
    /// ヘッダのフレーム番号（16 bit）
    pub counter: u64,
    /// 採用したヘッダの候補の順位（0 = 最尤）。0 でないとき、最尤の候補は CRC が合わなかった
    pub rank: usize,
    /// 最尤の候補との対数尤度の差 [nat]（0 = 最尤）。大きいほど、採用の根拠が弱い（下流の信頼度に使える）
    pub gap: f64,
}

fn val(bits: &[u8]) -> u64 {
    bits.iter().fold(0u64, |a, &b| (a << 1) | b as u64)
}

/// unpack の設定
#[derive(Clone, Copy)]
pub struct UnpackOptions<'a> {
    /// ヘッダの情報ビット（n_hdata 個）の事前の軟値（正 = 0 らしい）。ビタビ復号の情報ビット側に足す
    pub header_prior: Option<&'a [f64]>,
    /// ヘッダの候補の数（尤度の高い順。1 なら通常のビタビ復号と CRC 検査）。
    /// CRC は 12 bit なので、候補が多いほど、誤ったヘッダが偶然 CRC を通る確率（1 候補あたり 2^-12）が増える。8 なら、測定した範囲（Rayleigh 4〜8 dB まで）で誤採用は 0.2% 以下
    pub list: usize,
    /// Y 帯域の、専用シンボルの位相から得た（最上位ビットの軟値、パリティの軟値）。なければ消失として扱う
    pub y_soft: Option<(&'a [f64], &'a [f64])>,
}

impl Default for UnpackOptions<'_> {
    fn default() -> Self {
        UnpackOptions { header_prior: None, list: 8, y_soft: None }
    }
}

/// pack の逆。soft: 送信ビット列全体の軟値（自然対数の LLR、正 = 0 らしい）、bits: 硬判定。
/// ヘッダを尤度の高い順の候補にして（最尤が合わなければリストビタビ）、基本 CRC が最初に合うものを採る。合う候補がなければ None。
/// 純粋な関数。
pub fn unpack(lay: &Layout, soft: &[f64], bits: &[u8], opt: &UnpackOptions) -> Option<Unpacked> {
    use crate::codec::{qmax_of, qmin_of};
    let p = &lay.p;
    let pick = |idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| soft[i]).collect() };
    let mut header_soft = pick(&lay.hdata_idx);
    if let Some(pr) = opt.header_prior {
        for (x, p) in header_soft.iter_mut().zip(pr) {
            *x += p;
        }
    }
    let top: Vec<u8> = if lay.n_top > 0 {
        let t6 = fec::K - 1;
        let top_stream: Vec<usize> = lay.top_coef_idx.iter().map(|&i| lay.coef_stream_idx[i]).collect();
        let mut par = pick(&lay.parity_idx);
        par.extend(pick(&lay.tail_idx[..t6]));
        fec::rsc_decode(&pick(&top_stream), &par, &pick(&lay.tail_idx[t6..]))
    } else {
        Vec::new()
    };
    let ymsb: Vec<u8> = if lay.y_items.is_empty() {
        Vec::new()
    } else {
        let t6 = fec::K - 1;
        let n = lay.y_items.len();
        let zero = vec![0.0; n];
        let (msb, par) = opt.y_soft.unwrap_or((&zero, &zero));
        let mut parity = par.to_vec();
        parity.extend(pick(&lay.ytail_idx[..t6]));
        fec::rsc_decode(msb, &parity, &pick(&lay.ytail_idx[t6..]))
    };
    let n_core = fec::CRC_CORE.0;
    // 基本 CRC と固定値が合えば、高域 CRC の結果を返す
    let accept = |prot: &[u8]| -> Option<bool> {
        let body = &prot[..lay.n_prot];
        let mut core: Vec<u8> = lay.core_prot.iter().map(|&i| body[i]).collect();
        core.extend(lay.core_top.iter().map(|&i| top[i]));
        core.extend(lay.core_y.iter().map(|&i| ymsb[i]));
        if fec::crc_core(&core) != prot[lay.n_prot..lay.n_prot + n_core] {
            return None;
        }
        let hdr = &body[..lay.header_bits];
        if val(&hdr[0..4]) != p.pid as u64 || hdr[5] != 0 || val(&hdr[6..8]) != p.format_id as u64 {
            return None;
        }
        let mut high: Vec<u8> = lay.high_prot.iter().map(|&i| body[i]).collect();
        high.extend(lay.high_top.iter().map(|&i| top[i]));
        high.extend(lay.high_y.iter().map(|&i| ymsb[i]));
        Some(!lay.has_high_crc || fec::crc_high(&high) == prot[lay.n_prot + n_core..])
    };
    let (hpar, htail) = (pick(&lay.hpar_idx), pick(&lay.htail_idx));
    // 最尤（ビタビ）が通ればそれを採る。通らなければ、尤度の高い順の候補（リストビタビ）から最初に通るもの
    let best = fec::rsc_decode(&header_soft, &hpar, &htail);
    let (rank, gap, prot, high_ok) = match accept(&best) {
        Some(h) => (0, 0.0, best, h),
        None => {
            let candidates = fec::rsc_decode_list(&header_soft, &hpar, &htail, opt.list.max(1));
            let top_metric = candidates.first()?.1;
            candidates
                .into_iter()
                .enumerate()
                .skip(1)
                .find_map(|(rank, (prot, metric))| accept(&prot).map(|h| (rank, (top_metric - metric) / 2.0, prot, h)))?
        }
    };
    let body = &prot[..lay.n_prot];
    let hdr = &body[..lay.header_bits];
    let mut coef_bits: Vec<u8> = lay.coef_stream_idx.iter().map(|&i| bits[i]).collect();
    for (t, &ci) in lay.top_coef_idx.iter().enumerate() {
        coef_bits[ci] = top[t];
    }
    let nb = p.nbands();
    let mut i = lay.header_bits;
    let mut eidx = vec![vec![None; nb]; 2];
    let mut attn = vec![vec![None; nb]; 2];
    for &(ch, k) in &lay.exp_items {
        eidx[ch][k] = Some(val(&body[i..i + p.exp_bits]) as i64);
        i += p.exp_bits;
    }
    for &(ch, k) in &lay.exp_items {
        let count = p.attenuator_bits(ch, k);
        attn[ch][k] = Some(body[i..i + count].iter().map(|&b| b as i64).collect());
        i += count;
    }
    let mut u_flat = vec![0i64; lay.n_coef];
    for (j, &cf) in lay.coef_flat.iter().enumerate() {
        u_flat[cf] += (coef_bits[j] as i64) << lay.coef_pos[j];
    }
    for (t, &cf) in lay.y_flat.iter().enumerate() {
        let (ch, k, _) = lay.y_items[t];
        u_flat[cf] += (ymsb[t] as i64) << (p.bits(ch)[k] - 1);
    }
    let mut mant = vec![vec![None; nb]; 2];
    for &(ch, k) in &lay.coef_groups {
        let o = lay.coef_offset[&(ch, k)];
        let b = p.bits(ch)[k];
        mant[ch][k] =
            Some(u_flat[o..o + p.ncoef(k)].iter().map(|&u| (u - offset_of(b, p, ch, k)).clamp(qmin_of(b, p, ch, k), qmax_of(b))).collect());
    }
    Some(Unpacked {
        bd: BlockData { mant, eidx, attn },
        high_ok,
        top_bits: top,
        y_msb: ymsb,
        counter: val(&hdr[8..lay.header_bits]),
        rank,
        gap,
    })
}
