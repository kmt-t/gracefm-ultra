//! フレームのビット配置（GraceFM `gracefm/layout.py` の移植。暗号化は未対応）。

use std::collections::HashMap;

use crate::codec::offset_of;
use crate::fec;
use crate::profiles::Profile;

#[derive(Clone, Copy, Debug)]
enum Tok {
    P(usize),
    C(usize, usize, usize, i64), // ch, k, j, pos
    Hp(usize),
    Hd(usize),
    Ht(usize),
    Cp(usize),
    Ct(usize),
    Yp(usize),
    Yt(usize),
    Z,
}

pub struct Layout {
    pub p: Profile,
    pub header_bits: usize,
    pub exp_items: Vec<(usize, usize)>,
    pub attn_items: Vec<(usize, usize, usize)>,
    pub n_top: usize,
    pub n_prot: usize,
    pub core_prot: Vec<usize>,
    pub high_prot: Vec<usize>,
    pub core_top: Vec<usize>,
    pub high_top: Vec<usize>,
    pub has_high_crc: bool,
    pub n_hdata: usize,
    pub n_strong: usize,
    pub n_tx: usize,
    pub hpar_idx: Vec<usize>,
    pub hdata_idx: Vec<usize>,
    pub htail_idx: Vec<usize>,
    pub tail_idx: Vec<usize>,
    pub parity_idx: Vec<usize>,
    pub coef_stream_idx: Vec<usize>,
    pub top_coef_idx: Vec<usize>,
    pub coef_pos: Vec<i64>,
    pub coef_flat: Vec<usize>,
    /// 仮数ビットの係数の、成分ごとに全帯域を並べた位置（成分 ch の係数は 64 ch .. 64 ch + 63 番）。送らない帯域と成分の分も数える
    pub coef_full: Vec<usize>,
    pub coef_offset_bits: Vec<i64>,
    pub coef_groups: Vec<(usize, usize)>,
    pub coef_offset: HashMap<(usize, usize), usize>,
    pub n_coef: usize,
    /// Y 帯域の係数 (ch, k, j)（専用シンボルで送る。仮数ビットの列には入らない）
    pub y_items: Vec<(usize, usize, usize)>,
    pub y_flat: Vec<usize>,
    pub ytail_idx: Vec<usize>,
    pub core_y: Vec<usize>,
    pub high_y: Vec<usize>,
    /// 受信の観測のビット: 仮数ビットの列に続けて、Y 帯域の係数ごとに上位から全ビット。平坦な係数の番号、全帯域を並べた位置、仮数内の位置
    pub obs_flat: Vec<usize>,
    pub obs_full: Vec<usize>,
    pub obs_pos: Vec<i64>,
}

fn int_bits(v: u64, n: usize) -> Vec<u8> {
    (0..n).map(|i| ((v >> (n - 1 - i)) & 1) as u8).collect()
}

impl Layout {
    pub fn new(p: &Profile) -> Result<Layout, String> {
        let nb = p.nbands();
        let header_bits = 24;
        let mut exp_items = Vec::new();
        let mut attn_items = Vec::new();
        for ch in 0..2 {
            for k in 0..nb {
                if p.bits(ch)[k] > 0 {
                    exp_items.push((ch, k));
                    for j in 0..p.attenuator_bits(ch, k) {
                        attn_items.push((ch, k, j));
                    }
                }
            }
        }
        let mut strong: Vec<Tok> = Vec::new();
        let mut weak: Vec<Tok> = Vec::new();
        let mut top_k: Vec<usize> = Vec::new();
        let mut top_item: Vec<(usize, usize, usize, i64)> = Vec::new();
        let mut y_items = Vec::new();
        for ch in 0..2 {
            for k in 0..nb {
                let b = p.bits(ch)[k];
                if b == 0 {
                    continue;
                }
                let kk = p.protect_bits(k);
                let base = p.base_bits(ch, k);
                for j in 0..p.ncoef(k) {
                    if p.y_band(ch, k) {
                        y_items.push((ch, k, j));
                        continue;
                    }
                    let mut par = Vec::new();
                    for rank in 0..kk as i64 {
                        par.push(Tok::P(top_item.len()));
                        top_k.push(k);
                        top_item.push((ch, k, j, b - 1 - rank));
                    }
                    let dat: Vec<Tok> = (0..base).map(|rank| Tok::C(ch, k, j, b - 1 - rank)).collect();
                    for i in 0..par.len().max(dat.len()) {
                        if i < par.len() {
                            strong.push(par[i]);
                        }
                        if i < dat.len() {
                            strong.push(dat[i]);
                        }
                    }
                    for rank in base..b {
                        weak.push(Tok::C(ch, k, j, b - 1 - rank));
                    }
                }
            }
        }
        let n_top = top_item.len();
        let n_prot = header_bits + p.exp_bits * exp_items.len() + attn_items.len();
        let (mut core_prot, mut high_prot) = (Vec::new(), Vec::new());
        let mut core_top = Vec::new();
        let mut high_top = Vec::new();
        for i in 0..header_bits {
            core_prot.push(i);
        }
        let mut bit = header_bits;
        for &(_, k) in &exp_items {
            let dst = if k >= p.high_k() { &mut high_prot } else { &mut core_prot };
            dst.extend(bit..bit + p.exp_bits);
            bit += p.exp_bits;
        }
        for &(_, k, _) in &attn_items {
            let dst = if k >= p.high_k() { &mut high_prot } else { &mut core_prot };
            dst.push(bit);
            bit += 1;
        }
        assert_eq!(bit, n_prot);
        for (t, &k) in top_k.iter().enumerate() {
            if k < p.high_k() {
                core_top.push(t)
            } else {
                high_top.push(t)
            }
        }
        let (core_y, high_y): (Vec<usize>, Vec<usize>) = (0..y_items.len()).partition(|&t| y_items[t].1 < p.high_k());
        let has_high_crc = !high_prot.is_empty() || !high_top.is_empty() || !high_y.is_empty();
        let n_prot_crc = n_prot + fec::CRC_CORE.0 + if has_high_crc { fec::CRC_HIGH.0 } else { 0 };
        let t6 = fec::K - 1;
        let n_hdata = n_prot_crc;

        let mut strong_list: Vec<Tok> = (0..n_hdata + t6).map(Tok::Hp).collect();
        let mut weak_list: Vec<Tok> = (0..n_hdata).map(Tok::Hd).chain((0..t6).map(Tok::Ht)).collect();
        if n_top > 0 {
            strong_list.extend((0..t6).map(Tok::Cp));
            weak_list.extend((0..t6).map(Tok::Ct));
        }
        if !y_items.is_empty() {
            strong_list.extend((0..t6).map(Tok::Yp));
            weak_list.extend((0..t6).map(Tok::Yt));
        }
        strong_list.extend(strong);
        weak_list.extend(weak);
        let mut n_strong;
        if p.layered() {
            let length = 2 * strong_list.len().max(weak_list.len()).div_ceil(2);
            strong_list.resize(length, Tok::Z);
            weak_list.resize(length, Tok::Z);
            n_strong = length;
        } else {
            n_strong = 0;
        }
        let mut stream_list = strong_list;
        stream_list.extend(weak_list);
        if !p.layered() {
            n_strong = stream_list.len();
        }
        let n_tx = stream_list.len();

        let mut hpar_idx = vec![0; n_hdata + t6];
        let mut hdata_idx = vec![0; n_hdata];
        let mut htail_idx = vec![0; t6];
        let mut tail_idx = vec![0; if n_top > 0 { 2 * t6 } else { 0 }];
        let mut ytail_idx = vec![0; if y_items.is_empty() { 0 } else { 2 * t6 }];
        let mut parity_idx = vec![0; n_top];
        let mut coef_items = Vec::new();
        let mut coef_stream_idx = Vec::new();
        for (pos, tok) in stream_list.iter().enumerate() {
            match *tok {
                Tok::Hp(i) => hpar_idx[i] = pos,
                Tok::Hd(i) => hdata_idx[i] = pos,
                Tok::Ht(i) => htail_idx[i] = pos,
                Tok::Cp(i) => tail_idx[i] = pos,
                Tok::Ct(i) => tail_idx[t6 + i] = pos,
                Tok::Yp(i) => ytail_idx[i] = pos,
                Tok::Yt(i) => ytail_idx[t6 + i] = pos,
                Tok::P(i) => parity_idx[i] = pos,
                Tok::C(ch, k, j, bp) => {
                    coef_items.push((ch, k, j, bp));
                    coef_stream_idx.push(pos);
                }
                Tok::Z => {}
            }
        }
        let mut index_of = HashMap::new();
        for (i, it) in coef_items.iter().enumerate() {
            index_of.insert(*it, i);
        }
        let top_coef_idx: Vec<usize> = top_item.iter().map(|it| index_of[it]).collect();
        let mut coef_offset = HashMap::new();
        let mut coef_groups = Vec::new();
        let mut off = 0;
        for ch in 0..2 {
            for k in 0..nb {
                if p.bits(ch)[k] > 0 {
                    coef_offset.insert((ch, k), off);
                    coef_groups.push((ch, k));
                    off += p.ncoef(k);
                }
            }
        }
        let full_of = |ch: usize, k: usize, j: usize| ch * p.block + (0..k).map(|i| p.ncoef(i)).sum::<usize>() + j;
        let coef_pos: Vec<i64> = coef_items.iter().map(|t| t.3).collect();
        let coef_flat: Vec<usize> = coef_items.iter().map(|t| coef_offset[&(t.0, t.1)] + t.2).collect();
        let coef_full: Vec<usize> = coef_items.iter().map(|t| full_of(t.0, t.1, t.2)).collect();
        let coef_offset_bits = coef_items.iter().map(|t| offset_of(p.bits(t.0)[t.1], p, t.0, t.1)).collect();
        let y_flat: Vec<usize> = y_items.iter().map(|&(ch, k, j)| coef_offset[&(ch, k)] + j).collect();
        let (mut obs_flat, mut obs_full, mut obs_pos) = (coef_flat.clone(), coef_full.clone(), coef_pos.clone());
        for (t, &(ch, k, j)) in y_items.iter().enumerate() {
            for pos in (0..p.bits(ch)[k]).rev() {
                obs_flat.push(y_flat[t]);
                obs_full.push(full_of(ch, k, j));
                obs_pos.push(pos);
            }
        }
        Ok(Layout {
            p: p.clone(),
            header_bits,
            exp_items,
            attn_items,
            n_top,
            n_prot,
            core_prot,
            high_prot,
            core_top,
            high_top,
            has_high_crc,
            n_hdata,
            n_strong,
            n_tx,
            hpar_idx,
            hdata_idx,
            htail_idx,
            tail_idx,
            parity_idx,
            coef_stream_idx,
            top_coef_idx,
            coef_pos,
            coef_flat,
            coef_full,
            coef_offset_bits,
            coef_groups,
            coef_offset,
            n_coef: off,
            y_items,
            y_flat,
            ytail_idx,
            core_y,
            high_y,
            obs_flat,
            obs_full,
            obs_pos,
        })
    }

    /// 係数ごとの仮数 bit 数
    pub fn n_coef_bits(&self) -> usize {
        self.coef_stream_idx.len()
    }

    /// 受信の観測のビットの数
    pub fn n_obs_bits(&self) -> usize {
        self.obs_pos.len()
    }

    pub fn header_prefix(&self, stream_id: u64, counter: u64) -> Vec<u8> {
        let p = &self.p;
        let mut h = int_bits(p.pid as u64, 4);
        h.extend(int_bits(stream_id, 1));
        h.push(0);
        h.extend(int_bits(p.format_id as u64, 2));
        let counter_bits = self.header_bits - 8;
        h.extend(int_bits(counter & ((1u64 << counter_bits) - 1), counter_bits));
        h
    }

    pub fn exp_bits_of(&self, v: i64) -> Vec<u8> {
        int_bits(v as u64, self.p.exp_bits)
    }
}
