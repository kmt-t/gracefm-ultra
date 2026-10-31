//! フレームの送信（GraceFM `gracefm/framing.py` の pack の移植。暗号化なし）。

use crate::codec::BlockData;
use crate::fec;
use crate::layout::Layout;

/// BlockData → 送信ビット列（n_tx 個、スクランブル前）
pub fn pack(lay: &Layout, bd: &BlockData, stream_id: u64, counter: u64) -> Vec<u8> {
    let mut prot = lay.header_prefix(stream_id, counter);
    for &(ch, k) in &lay.exp_items {
        prot.extend(lay.exp_bits_of(bd.eidx[ch][k].unwrap()));
    }
    for &(ch, k, j) in &lay.attn_items {
        prot.push(bd.attn[ch][k].as_ref().unwrap()[j] as u8);
    }
    let mut flat_mant: Vec<i64> = Vec::with_capacity(lay.n_coef);
    for &(ch, k) in &lay.coef_groups {
        flat_mant.extend(bd.mant[ch][k].as_ref().unwrap());
    }
    let coef: Vec<u8> = (0..lay.n_coef_bits())
        .map(|i| (((flat_mant[lay.coef_flat[i]] + lay.coef_offset_bits[i]) >> lay.coef_pos[i]) & 1) as u8)
        .collect();
    let top: Vec<u8> = lay.top_coef_idx.iter().map(|&i| coef[i]).collect();
    let mut core: Vec<u8> = lay.core_prot.iter().map(|&i| prot[i]).collect();
    core.extend(lay.core_top.iter().map(|&i| top[i]));
    let mut high: Vec<u8> = lay.high_prot.iter().map(|&i| prot[i]).collect();
    high.extend(lay.high_top.iter().map(|&i| top[i]));
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
    for (i, &v) in coef.iter().enumerate() {
        stream[lay.coef_stream_idx[i]] = v;
    }
    stream
}

// ---------------------------------------------------------------- 受信

/// unpack の出力。protected_bits は FEC で訂正した保護ビット（上位 K bit）で、受信側が事後確率を確定値にするのに使う
pub struct Unpacked {
    pub bd: BlockData,
    pub high_ok: bool,
    pub protected_bits: Vec<u8>,
}

fn val(bits: &[u8]) -> u64 {
    bits.iter().fold(0u64, |a, &b| (a << 1) | b as u64)
}

/// pack の逆。soft: 送信ビット列全体の軟値（自然対数の LLR、正 = 0 らしい）、bits: 硬判定。基本 CRC が合わなければ None。純粋な関数
pub fn unpack(lay: &Layout, soft: &[f64], bits: &[u8]) -> Option<Unpacked> {
    use crate::codec::{offset_of, qmax_of, qmin_of};
    let p = &lay.p;
    let pick = |idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| soft[i]).collect() };
    let prot = fec::rsc_decode(&pick(&lay.hdata_idx), &pick(&lay.hpar_idx), &pick(&lay.htail_idx));
    let body = &prot[..lay.n_prot];
    let top: Vec<u8> = if lay.n_top > 0 {
        let t6 = fec::K - 1;
        let top_stream: Vec<usize> = lay.top_coef_idx.iter().map(|&i| lay.coef_stream_idx[i]).collect();
        let mut par = pick(&lay.parity_idx);
        par.extend(pick(&lay.tail_idx[..t6]));
        fec::rsc_decode(&pick(&top_stream), &par, &pick(&lay.tail_idx[t6..]))
    } else {
        Vec::new()
    };
    let mut core: Vec<u8> = lay.core_prot.iter().map(|&i| body[i]).collect();
    core.extend(lay.core_top.iter().map(|&i| top[i]));
    let mut high: Vec<u8> = lay.high_prot.iter().map(|&i| body[i]).collect();
    high.extend(lay.high_top.iter().map(|&i| top[i]));
    let n_core = fec::CRC_CORE.0;
    if fec::crc_core(&core) != prot[lay.n_prot..lay.n_prot + n_core] {
        return None;
    }
    let high_ok = !lay.has_high_crc || fec::crc_high(&high) == prot[lay.n_prot + n_core..];
    let hdr = &body[..lay.header_bits];
    if val(&hdr[0..4]) != p.pid as u64 || hdr[5] != 0 || val(&hdr[6..8]) != p.format_id as u64 {
        return None;
    }
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
    let mut mant = vec![vec![None; nb]; 2];
    for &(ch, k) in &lay.coef_groups {
        let o = lay.coef_offset[&(ch, k)];
        let b = p.bits(ch)[k];
        mant[ch][k] = Some(u_flat[o..o + p.ncoef(k)].iter().map(|&u| (u - offset_of(b)).clamp(qmin_of(b), qmax_of(b))).collect());
    }
    Some(Unpacked { bd: BlockData { mant, eidx, attn }, high_ok, protected_bits: top })
}

pub const RECOVER_CANDIDATES: usize = 7;
pub const RECOVER_MAX_FLIPS: usize = 2;

/// 基本 CRC が合わないとき、ヘッダの信頼度の低い bit を 1〜RECOVER_MAX_FLIPS 個反転して CRC が合う候補を探す（Chase 型）。
/// 返り値の 2 つ目は反転した個数
pub fn unpack_recovering(lay: &Layout, soft: &[f64], bits: &[u8]) -> Option<(Unpacked, usize)> {
    if let Some(r) = unpack(lay, soft, bits) {
        return Some((r, 0));
    }
    let mut hdr_idx: Vec<usize> = lay.hdata_idx.clone();
    hdr_idx.extend(&lay.hpar_idx);
    let mut order: Vec<usize> = (0..hdr_idx.len()).collect();
    order.sort_by(|&a, &b| soft[hdr_idx[a]].abs().partial_cmp(&soft[hdr_idx[b]].abs()).unwrap()); // 安定ソート
    let cand: Vec<usize> = order.iter().take(RECOVER_CANDIDATES).map(|&i| hdr_idx[i]).collect();
    for t in 1..=RECOVER_MAX_FLIPS {
        let mut comb: Vec<usize> = (0..t).collect();
        loop {
            let mut s2 = soft.to_vec();
            let mut b2 = bits.to_vec();
            for &c in &comb {
                let idx = cand[c];
                s2[idx] = -s2[idx];
                b2[idx] ^= 1;
            }
            if let Some(r) = unpack(lay, &s2, &b2) {
                return Some((r, t));
            }
            // 次の組み合わせ（辞書順）
            let mut i = t;
            let mut done = true;
            while i > 0 {
                i -= 1;
                if comb[i] != i + cand.len() - t {
                    comb[i] += 1;
                    for j in i + 1..t {
                        comb[j] = comb[j - 1] + 1;
                    }
                    done = false;
                    break;
                }
            }
            if done {
                break;
            }
        }
    }
    None
}
