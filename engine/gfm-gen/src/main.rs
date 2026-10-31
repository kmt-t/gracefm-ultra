//! 学習データ生成。使い方:
//!   gfm-gen --profile B200 --pcm clip.raw --scen e0,fade,rate,burst [--scen ...] --seed 1 --out out.npz [--threads 2] [--full] [--start N]
//! clip.raw: 符号つき 24 bit を int32 で入れた L/R の交互 (リトルエンディアン)。

use gfm_core::gen::{generate, Scen};
use gfm_core::profiles::profile_by_name;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut profile, mut pcm_path, mut out, mut seed, mut threads, mut full, mut start) =
        (String::new(), String::new(), String::new(), 0u64, 1usize, false, 0i64);
    let mut scens = Vec::new();
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().unwrap_or_else(|| panic!("{} に値がありません", args[i]));
        match args[i].as_str() {
            "--profile" => { profile = next(i); i += 2 }
            "--pcm" => { pcm_path = next(i); i += 2 }
            "--out" => { out = next(i); i += 2 }
            "--seed" => { seed = next(i).parse().unwrap(); i += 2 }
            "--threads" => { threads = next(i).parse().unwrap(); i += 2 }
            "--start" => { start = next(i).parse().unwrap(); i += 2 }
            "--full" => { full = true; i += 1 }
            "--scen" => {
                let v: Vec<f64> = next(i).split(',').map(|x| x.parse().unwrap()).collect();
                scens.push(Scen { e0: v[0], fade: v[1] != 0.0, rate: v[2], burst: v[3] });
                i += 2
            }
            a => panic!("不明な引数: {}", a),
        }
    }
    let p = profile_by_name(&profile).unwrap_or_else(|| panic!("不明なプロファイル: {}", profile));
    let raw = std::fs::read(&pcm_path).expect("pcm を読めません");
    let pcm: Vec<[i64; 2]> = raw
        .chunks_exact(8)
        .map(|c| [i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64, i32::from_le_bytes([c[4], c[5], c[6], c[7]]) as i64])
        .collect();
    let t0 = std::time::Instant::now();
    let o = generate(&p, &pcm, &scens, seed, threads, full).unwrap_or_else(|e| panic!("{}", e));
    o.write(&out, start).expect("書き出しに失敗");
    eprintln!("{}: {} frames x {} scenarios, {:.1}s", out, o.nfr, scens.len(), t0.elapsed().as_secs_f64());
}
