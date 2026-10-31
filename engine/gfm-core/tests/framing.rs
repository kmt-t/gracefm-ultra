//! フレーム化（pack / unpack）と、量子化結果の往復。

mod common;
use common::{stereo, PROFILES};
use gfm_core::codec::encode_block;
use gfm_core::framing::*;
use gfm_core::layout::Layout;
use gfm_core::profiles::profile_by_name;

fn soft_of(stream: &[u8], a: f64) -> Vec<f64> {
    stream.iter().map(|&b| if b == 0 { a } else { -a }).collect()
}

#[test]
fn pack_unpack_round_trip() {
    for name in PROFILES {
        let p = profile_by_name(name).unwrap();
        let lay = Layout::new(&p).unwrap();
        let x = stereo(64 * 8, 5);
        for (i, blk) in x.chunks(64).enumerate() {
            let bd = encode_block(blk, &p, true);
            let stream = pack(&lay, &bd, 0, i as u64);
            assert_eq!(stream.len(), lay.n_tx);
            let u = unpack(&lay, &soft_of(&stream, 20.0), &stream).unwrap_or_else(|| panic!("{}: CRC が通らない", name));
            assert!(u.high_ok, "{}", name);
            assert_eq!(u.bd.mant, bd.mant, "{}: 仮数", name);
            assert_eq!(u.bd.eidx, bd.eidx, "{}: 指数", name);
            assert_eq!(u.bd.attn, bd.attn, "{}: アッテネータ", name);
        }
    }
}

#[test]
fn header_of_another_profile_is_rejected() {
    let (a, b) = (profile_by_name("B200").unwrap(), profile_by_name("B200M").unwrap());
    let (lay_a, lay_b) = (Layout::new(&a).unwrap(), Layout::new(&b).unwrap());
    let bd = encode_block(&stereo(64, 1), &a, true);
    let stream = pack(&lay_a, &bd, 0, 0);
    assert!(unpack(&lay_a, &soft_of(&stream, 20.0), &stream).is_some());
    // 別のプロファイルの配置で読むと、プロファイル ID か CRC が合わない
    let mut s2 = stream.clone();
    s2.resize(lay_b.n_tx.max(stream.len()), 0);
    assert!(unpack(&lay_b, &soft_of(&s2, 20.0)[..lay_b.n_tx], &s2[..lay_b.n_tx]).is_none());
}

#[test]
fn heavy_corruption_fails_the_crc() {
    let p = profile_by_name("B200").unwrap();
    let lay = Layout::new(&p).unwrap();
    let bd = encode_block(&stereo(64, 2), &p, true);
    let stream = pack(&lay, &bd, 0, 0);
    // ヘッダ部をすべて逆向きの強い軟値にする
    let mut soft = soft_of(&stream, 20.0);
    for &i in lay.hdata_idx.iter().chain(&lay.hpar_idx) {
        soft[i] = -soft[i];
    }
    let hard: Vec<u8> = soft.iter().map(|&s| (s < 0.0) as u8).collect();
    assert!(unpack_recovering(&lay, &soft, &hard).is_none());
}

#[test]
fn recovery_returns_the_plain_result_when_the_frame_is_fine() {
    let p = profile_by_name("M700").unwrap();
    let lay = Layout::new(&p).unwrap();
    let bd = encode_block(&stereo(64, 3), &p, true);
    let stream = pack(&lay, &bd, 1, 7);
    let (u, flips) = unpack_recovering(&lay, &soft_of(&stream, 10.0), &stream).unwrap();
    assert_eq!(flips, 0);
    assert_eq!(u.bd.mant, bd.mant);
}

#[test]
fn coefficient_bits_reassemble_the_mantissa() {
    let p = profile_by_name("B200").unwrap();
    let lay = Layout::new(&p).unwrap();
    let bd = encode_block(&stereo(64, 4), &p, true);
    let bits = coefficient_bits(&lay, &bd);
    assert_eq!(bits.len(), lay.n_coef_bits());
    // ビット位置の重みつき和 - オフセット = 仮数
    let mut u = vec![0i64; lay.n_coef];
    for (j, &cf) in lay.coef_flat.iter().enumerate() {
        u[cf] += (bits[j] as i64) << lay.coef_pos[j];
    }
    for &(ch, k) in &lay.coef_groups {
        let o = lay.coef_offset[&(ch, k)];
        let b = p.bits(ch)[k];
        let m: Vec<i64> = u[o..o + p.ncoef(k)].iter().map(|v| v - gfm_core::codec::offset_of(b)).collect();
        assert_eq!(&m, bd.mant[ch][k].as_ref().unwrap());
    }
}
