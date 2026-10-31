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
