//! CRC 付きリスト復号（`unpack` の list）の効果と危険: 基本 CRC が通らない割合（fail）と、
//! 採用したが、ヘッダが真値と違う割合（wrong、誤採用）を、設定ごとに測る。CRC が合っても正しい保証はない。
//!   cargo run --release --example crc_list -- --profile B200 --pcm clip.raw
//! 設定: list（ヘッダの候補数）、prior（ヘッダの事前知識）。

use gfm_core::channel::{awgn_adc, Link, Scen};
use gfm_core::codec::BlockData;
use gfm_core::encoder::Encoder;
use gfm_core::framing::{unpack, UnpackOptions};
use gfm_core::modem::{Modem, N_PREAMBLE};
use gfm_core::pcm::read_pcm24_le;
use gfm_core::prior::{PriorConfig, Tracker};
use gfm_core::profiles::profile_by_name;
use gfm_core::scrambler::Scrambler;

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

    // (名前, list, prior)
    let configs: [(&str, usize, bool); 6] = [
        ("L1", 1, false),
        ("L1+事前", 1, true),
        ("L4+事前", 4, true),
        ("L8+事前", 8, true),
        ("L16+事前", 16, true),
        ("L32+事前", 32, true),
    ];
    println!("{}: 基本 CRC 不一致 fail [%] / 誤採用 wrong [%]（{} フレーム）", p.name, nfr);
    println!("{:16}{}", "通信路", configs.iter().map(|c| format!("{:>16}", c.0)).collect::<String>());
    for (kind, fade) in [("awgn", false), ("rayleigh", true)] {
        for e0 in [4.0, 5.0, 6.0, 8.0] {
            let mut cells = Vec::new();
            for &(_, list, use_prior) in &configs {
                let mut link = Link::new(&Scen { e0, fade, rate: 0.0, burst: 1.0 }, nfr, 1, 0);
                let mut tracker = Tracker::new();
                let (mut fail, mut wrong) = (0usize, 0usize);
                for (i, (bd, _, sym)) in frames.iter().enumerate() {
                    let rx = awgn_adc(sym, link.esn0[i], &mut link.noise);
                    let (mut llr, mut hard, _) = modem.demodulate(&rx[N_PREAMBLE..], lay.n_tx);
                    scr.descramble(&mut llr, &mut hard);
                    let prior = use_prior.then(|| tracker.header_prior(&lay, i, &PriorConfig::FULL));
                    let opt = UnpackOptions { header_prior: prior.as_deref(), list, ..UnpackOptions::default() };
                    match unpack(&lay, &llr, &hard, &opt) {
                        None => fail += 1,
                        Some(u) => {
                            // 誤採用: 基本 CRC が守る項目（フレーム番号、k < high_k の指数・アッテネータ）が違う。
                            // 高域（k >= high_k）は高域 CRC が守るので、high_ok のときだけ比べる（high_ok でなければ、高域は捨てる）
                            let bands = if u.high_ok { p.nbands() } else { p.high_k() };
                            let header_differs = (0..2)
                                .any(|ch| (0..bands).any(|k| u.bd.eidx[ch][k] != bd.eidx[ch][k] || u.bd.attn[ch][k] != bd.attn[ch][k]));
                            if u.counter != (i as u64 & 0xFFFF) || header_differs {
                                wrong += 1;
                            }
                            tracker.observe(i, u.counter, &u.bd);
                        }
                    }
                }
                cells.push(format!("{:>9.1} /{:>5.2}", 100.0 * fail as f64 / nfr as f64, 100.0 * wrong as f64 / nfr as f64));
            }
            println!("{:16}{}", format!("{} {} dB", kind, e0), cells.iter().map(|c| format!("{:>16}", c)).collect::<String>());
        }
    }
}
