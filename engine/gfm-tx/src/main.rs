//! 送信と通信路（評価用）。使い方:
//!   gfm-tx --profile B200 --pcm clip.raw --scen e0,fade,rate,burst --seed 1 --out a.sym
//! clip.raw: 符号つき 24 bit を int32 で入れた L/R の交互（リトルエンディアン、満点 = 2^23）。
//! a.sym: u32 フレーム数, u32 フレームあたりのシンボル数, フレーム数個のフラグ (u8、1 = 届いた), 届いたフレームの I/Q (f32 リトルエンディアン、±1 = 1.0)。

use gfm_core::channel::{awgn_adc, Link, Scen};
use gfm_core::encoder::Encoder;
use gfm_core::pcm::read_pcm24_le;
use gfm_core::profiles::profile_by_name;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut profile, mut pcm_path, mut out) = (String::new(), String::new(), String::new());
    let mut seed = 0u64;
    let mut scen = Scen { e0: 30.0, fade: false, rate: 0.0, burst: 1.0 };
    let mut i = 1;
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_else(|| panic!("{} に値がありません", args[i]));
        match args[i].as_str() {
            "--profile" => profile = next,
            "--pcm" => pcm_path = next,
            "--out" => out = next,
            "--seed" => seed = next.parse().unwrap(),
            "--scen" => {
                let v: Vec<f64> = next.split(',').map(|x| x.parse().unwrap()).collect();
                scen = Scen { e0: v[0], fade: v[1] != 0.0, rate: v[2], burst: v[3] };
            }
            a => panic!("不明な引数: {}", a),
        }
        i += 2;
    }
    let p = profile_by_name(&profile).unwrap_or_else(|| panic!("不明なプロファイル: {}", profile));
    let raw = std::fs::read(&pcm_path).expect("pcm を読めません");
    let pcm = read_pcm24_le(&raw);
    let frames = Encoder::new(&p).unwrap_or_else(|e| panic!("{}", e)).run(&pcm);
    let mut link = Link::new(&scen, frames.len(), seed, 0);
    let ns = frames[0].len();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend((frames.len() as u32).to_le_bytes());
    buf.extend((ns as u32).to_le_bytes());
    buf.extend(link.lost.iter().map(|&l| if l { 0u8 } else { 1u8 }));
    for (i, f) in frames.iter().enumerate() {
        if link.lost[i] {
            continue;
        }
        for s in awgn_adc(f, link.esn0[i], &mut link.noise) {
            buf.extend((s[0] as f32).to_le_bytes());
            buf.extend((s[1] as f32).to_le_bytes());
        }
    }
    std::fs::write(&out, buf).expect("書き出しに失敗");
}
