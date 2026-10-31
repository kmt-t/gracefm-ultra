//! 事前知識の効果の上限: ヘッダや保護ビットを「正解」で事前に与えたときの、基本 CRC の不一致の割合。
//! 実際の予測（フレーム番号、指数）が、どこまで近づけるかの目安になる。
//!   cargo run --release --example prior_bound -- --profile B200 --pcm clip.raw
//! 列: none（通信路の軟値だけ）、予測できる項目（固定ビット、フレーム番号、指数）を正解で、ヘッダ全体（アッテネータと CRC も）を正解で、
//!     保護ビット（仮数の上位）を正解で、それらの組み合わせ。

use gfm_core::channel::{awgn_adc, Link, Scen};
use gfm_core::codec::BlockData;
use gfm_core::encoder::Encoder;
use gfm_core::framing::{coefficient_bits, unpack_recovering, UnpackPrior};
use gfm_core::modem::{Modem, N_PREAMBLE};
use gfm_core::pcm::read_pcm24_le;
use gfm_core::profiles::profile_by_name;
use gfm_core::scrambler::Scrambler;

const STRONG: f64 = 12.0;

fn prior_of(bits: &[u8]) -> Vec<f64> {
    bits.iter().map(|&b| if b == 0 { STRONG } else { -STRONG }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let p = profile_by_name(&arg("--profile").expect("--profile")).expect("不明なプロファイル");
    let pcm = read_pcm24_le(&std::fs::read(arg("--pcm").expect("--pcm")).expect("pcm を読めません"));
    let mut enc = Encoder::new(&p, 0, 0).unwrap();
    let lay = gfm_core::layout::Layout::new(&p).unwrap();
    let modem = Modem::new(p.qam16h, p.alpha);
    let scr = Scrambler::new(modem.n_symbols(lay.n_tx, lay.n_strong) * modem.bits_per_symbol());
    let nfr = pcm.len() / p.block;
    let frames: Vec<(BlockData, Vec<u8>, Vec<[f64; 2]>)> =
        (0..nfr).map(|i| enc.frame_parts(&pcm[i * p.block..(i + 1) * p.block])).collect();

    // 正解の事前の軟値
    let n_fixed = lay.header_bits + lay.exp_items.len() * p.exp_bits; // ヘッダ 24 bit + 指数
    let truth: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = frames
        .iter()
        .map(|(bd, stream, _)| {
            let hdr: Vec<u8> = lay.hdata_idx.iter().map(|&i| stream[i]).collect();
            let mut predictable = prior_of(&hdr);
            predictable[n_fixed..].fill(0.0); // アッテネータと CRC は予測しない
            let coef = coefficient_bits(&lay, bd);
            let top: Vec<u8> = lay.top_coef_idx.iter().map(|&i| coef[i]).collect();
            (predictable, prior_of(&hdr), prior_of(&top))
        })
        .collect();

    let names = ["none", "予測できる項目", "ヘッダ全体", "保護ビット", "予測+保護", "ヘッダ+保護"];
    println!("{} の基本 CRC 不一致 [%]（{} フレーム）", p.name, nfr);
    println!("{:16}{}", "通信路", names.iter().map(|n| format!("{:>14}", n)).collect::<String>());
    for (kind, fade) in [("awgn", false), ("rayleigh", true)] {
        for e0 in [4.0, 6.0, 8.0] {
            let mut link = Link::new(&Scen { e0, fade, rate: 0.0, burst: 1.0 }, nfr, 1, 0);
            let mut fails = [0usize; 6];
            let mut total = 0usize;
            for (i, (_, _, sym)) in frames.iter().enumerate() {
                let rx = awgn_adc(sym, link.esn0[i], &mut link.noise);
                let (mut llr, mut hard, _) = modem.demodulate(&rx[N_PREAMBLE..], lay.n_tx);
                scr.descramble(&mut llr, &mut hard);
                let (pr, hd, tp) = (&truth[i].0, &truth[i].1, &truth[i].2);
                let configs = [
                    UnpackPrior::default(),
                    UnpackPrior { header: Some(pr), top: None },
                    UnpackPrior { header: Some(hd), top: None },
                    UnpackPrior { header: None, top: Some(tp) },
                    UnpackPrior { header: Some(pr), top: Some(tp) },
                    UnpackPrior { header: Some(hd), top: Some(tp) },
                ];
                for (k, c) in configs.iter().enumerate() {
                    fails[k] += unpack_recovering(&lay, &llr, &hard, c).is_none() as usize;
                }
                total += 1;
            }
            println!(
                "{:16}{}",
                format!("{} {} dB", kind, e0),
                fails.iter().map(|&f| format!("{:>14.1}", 100.0 * f as f64 / total as f64)).collect::<String>()
            );
        }
    }
}
