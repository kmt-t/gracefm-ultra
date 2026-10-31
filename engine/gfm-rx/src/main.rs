//! 受信（デコーダ）。使い方:
//!   gfm-rx [--float] --profile B200 [--profile-b SUB100] --frames-a a.sym [--frames-b b.sym] --n N [--offset-b K] --out out.raw [--quality q.txt]
//! フレームファイル (.sym): u32 フレーム数, u32 フレームあたりのシンボル数, フレーム数個のフラグ (u8、1 = 届いた), 届いたフレームの I/Q (i32 リトルエンディアン)。
//! out.raw: L/R の交互の i32（符号つき 24 bit PCM）。--float のときは、浮動小数点の受信で、L/R の交互の f32（満点 ±1）。

use gfm_core::profiles::profile_by_name;
use gfm_core::float_rx::{receive_f, sym_to_f};
use gfm_core::receiver::receive;

fn read_sym(path: &str) -> Vec<Option<Vec<[i64; 2]>>> {
    let d = std::fs::read(path).unwrap_or_else(|_| panic!("{} を読めません", path));
    let u32at = |o: usize| u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as usize;
    let (nf, ns) = (u32at(0), u32at(4));
    let flags = &d[8..8 + nf];
    let mut o = 8 + nf;
    let mut out = Vec::with_capacity(nf);
    for &f in flags {
        if f == 0 {
            out.push(None);
            continue;
        }
        let mut fr = Vec::with_capacity(ns);
        for _ in 0..ns {
            let i = i32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as i64;
            let q = i32::from_le_bytes([d[o + 4], d[o + 5], d[o + 6], d[o + 7]]) as i64;
            fr.push([i, q]);
            o += 8;
        }
        out.push(Some(fr));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut profile, mut profile_b, mut fa, mut fb, mut out, mut qpath) = (String::new(), String::new(), String::new(), String::new(), String::new(), String::new());
    let (mut n, mut offset_b) = (0usize, 0usize);
    let mut float = false;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--float" {
            float = true;
            i += 1;
            continue;
        }
        let next = args.get(i + 1).cloned().unwrap_or_else(|| panic!("{} に値がありません", args[i]));
        match args[i].as_str() {
            "--profile" => profile = next,
            "--profile-b" => profile_b = next,
            "--frames-a" => fa = next,
            "--frames-b" => fb = next,
            "--out" => out = next,
            "--quality" => qpath = next,
            "--n" => n = next.parse().unwrap(),
            "--offset-b" => offset_b = next.parse().unwrap(),
            a => panic!("不明な引数: {}", a),
        }
        i += 2;
    }
    let p = profile_by_name(&profile).unwrap_or_else(|| panic!("不明なプロファイル: {}", profile));
    let pb = if profile_b.is_empty() { None } else { Some(profile_by_name(&profile_b).unwrap_or_else(|| panic!("不明なプロファイル: {}", profile_b))) };
    let frames_a = read_sym(&fa);
    let frames_b = if fb.is_empty() { None } else { Some(read_sym(&fb)) };
    let t0 = std::time::Instant::now();
    if float {
        let conv = |f: &Vec<Option<Vec<[i64; 2]>>>| -> Vec<Option<Vec<[f64; 2]>>> { f.iter().map(|x| x.as_ref().map(|v| sym_to_f(v))).collect() };
        let fa_f = conv(&frames_a);
        let fb_f = frames_b.as_ref().map(conv);
        let r = receive_f(&p, &fa_f, fb_f.as_deref(), n, offset_b, pb.as_ref()).unwrap_or_else(|e| panic!("{}", e));
        let mut raw = Vec::with_capacity(r.pcm.len() * 8);
        for s in &r.pcm {
            raw.extend((s[0] as f32).to_le_bytes());
            raw.extend((s[1] as f32).to_le_bytes());
        }
        std::fs::write(&out, raw).expect("書き出しに失敗");
        if !qpath.is_empty() {
            std::fs::write(&qpath, r.quality.iter().map(|q| q.to_string()).collect::<Vec<_>>().join("\n")).unwrap();
        }
        eprintln!(
            "[float] frames {} lost {} crc_fail {} high_crc_fail {} recovered {} plc_samples {} ({:.2}s)",
            r.stats_a.frames, r.stats_a.lost, r.stats_a.crc_fail, r.stats_a.high_crc_fail, r.stats_a.recovered, r.plc_samples, t0.elapsed().as_secs_f64()
        );
        return;
    }
    let r = receive(&p, &frames_a, frames_b.as_deref(), n, offset_b, pb.as_ref()).unwrap_or_else(|e| panic!("{}", e));
    let mut raw = Vec::with_capacity(r.pcm.len() * 8);
    for s in &r.pcm {
        raw.extend((s[0] as i32).to_le_bytes());
        raw.extend((s[1] as i32).to_le_bytes());
    }
    std::fs::write(&out, raw).expect("書き出しに失敗");
    if !qpath.is_empty() {
        std::fs::write(&qpath, r.quality.iter().map(|q| q.to_string()).collect::<Vec<_>>().join("\n")).unwrap();
    }
    eprintln!(
        "frames {} lost {} crc_fail {} high_crc_fail {} recovered {} plc_samples {} ({:.2}s)",
        r.stats_a.frames, r.stats_a.lost, r.stats_a.crc_fail, r.stats_a.high_crc_fail, r.stats_a.recovered, r.plc_samples, t0.elapsed().as_secs_f64()
    );
}
