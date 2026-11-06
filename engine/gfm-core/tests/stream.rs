//! 受信の流れ（補間 → 判定 → 後段）の、窓と区間の受け渡し。網の代わりに、入力をそのまま返す模型を使う。

use gfm_core::decoder::{BandVar, DecodedFrame, Status};
use gfm_core::gate::{self, Gate};
use gfm_core::layout::Layout;
use gfm_core::post::{PostFilter, PostInput, PostModel, PostOutput, BINS, LOOKAHEAD, MDCT_HOP, REACH};
use gfm_core::profiles::profile_by_name;
use gfm_core::restore::{MendBatch, MendModel, Mender, Observation, Restored, HOP, NC, WIN};

/// e0 = 復号した係数、sd = 0
struct EchoMend;

impl MendModel for EchoMend {
    fn infer(&mut self, w: &MendBatch) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((w.coef.clone(), vec![0.0; w.coef.len()]))
    }
}

/// 音声 = e0（係数をそのまま標本とみなす）。音声 [256(k − 1), …) は、入力のフレーム [4k − 4, …) と同じ位置
struct EchoPost;

const DIMS: [(usize, usize); 2] = [(3, 1), (2, 4)];

impl PostModel for EchoPost {
    fn state_dims(&self) -> Vec<(usize, usize)> {
        DIMS.to_vec()
    }

    fn step(&mut self, x: &PostInput) -> Result<PostOutput, String> {
        let l = x.frames - LOOKAHEAD;
        let n = (MDCT_HOP * (x.frames + 1) + 2 * REACH) * NC;
        let audio = (0..2).flat_map(|ch| x.e0[ch * n + REACH * NC..ch * n + REACH * NC + l * BINS].iter().copied()).collect();
        let u = DIMS.iter().map(|&(c, w)| vec![0.0; 2 * c * (w + x.frames)]).collect();
        Ok(PostOutput { audio, y: vec![0.0; 2 * BINS * l], u })
    }
}

/// フレーム i の係数がすべて i（M）と −i（S）のフレーム
fn frame(i: usize, nb: usize) -> DecodedFrame {
    let p = profile_by_name("B200-L").unwrap();
    let coef = (0..2).map(|ch| (0..nb).map(|k| vec![if ch == 0 { i as f64 } else { -(i as f64) }; p.ncoef(k)]).collect()).collect();
    let var: BandVar = Default::default();
    DecodedFrame { status: Status::Ok, coef, var, llr: Vec::new(), header: None, gap: 0.0, hint: None }
}

#[test]
fn mender_emits_every_frame_once_in_order() {
    let p = profile_by_name("B200-L").unwrap();
    let lay = Layout::new(&p).unwrap();
    let mut m = Mender::new(&lay, EchoMend);
    let n = 437;
    let mut got: Vec<f32> = Vec::new();
    let check = |out: gfm_core::restore::Mended, got: &mut Vec<f32>| {
        for (j, r) in out.ahead.iter().enumerate() {
            assert_eq!(r.e0[0], (got.len() + out.fin.len() + j) as f32, "暫定のフレームは確定の続き");
        }
        got.extend(out.fin.iter().map(|r| r.e0[0]));
    };
    for i in 0..n {
        if let Some(out) = m.push(Observation::new(&lay, &frame(i, p.nbands()))).unwrap() {
            assert!((i + 1 - WIN).is_multiple_of(HOP), "窓は HOP ごと");
            check(out, &mut got);
        }
    }
    check(m.finish().unwrap().unwrap(), &mut got);
    assert_eq!(got, (0..n).map(|i| i as f32).collect::<Vec<_>>());
}

#[test]
fn post_filter_output_matches_the_input_frames() {
    let mut pf = PostFilter::new(EchoPost);
    let r = |i: usize| Restored { e0: [vec![i as f32; NC], vec![0.5; NC]].concat(), sd: vec![0.0; 2 * NC], usable: true };
    let mut out = Vec::new();
    let mut done = 0;
    for len in [75, 50, 50, 50, 50, 50, 50, 50, 50, 50, 37] {
        let fin = (done..done + len).map(|i| (r(i), true)).collect();
        let ahead: Vec<Restored> = (done + len..done + len + 25).map(r).collect();
        out.extend(pf.push(fin, &ahead).unwrap());
        done += len;
    }
    assert_eq!(out.len(), done * NC);
    for (s, v) in out.iter().enumerate() {
        let i = (s / NC) as f64;
        assert_eq!(*v, [i + 0.5, i - 0.5], "標本 {s}");
    }
}

#[test]
fn gate_turns_on_after_a_second_with_hysteresis() {
    let mut g = Gate::new(&profile_by_name("B200-L").unwrap());
    let r = |snr_db: f64| Restored { e0: vec![1.0; 2 * NC], sd: vec![10f64.powf(-snr_db / 20.0) as f32; 2 * NC], usable: true };
    for _ in 0..gate::WIN {
        assert!(!g.push(&r(40.0)), "過去が揃うまではオフ");
    }
    assert!(g.push(&r(40.0)));
    // OFF と ON の間ではオンのまま
    let mid = (gate::ON + gate::OFF) / 2.0;
    for _ in 0..2 * gate::WIN {
        assert!(g.push(&r(mid)));
    }
    let mut t = 0;
    while g.push(&r(gate::OFF - 3.0)) {
        t += 1;
    }
    assert!(t < gate::WIN);
    for _ in 0..2 * gate::WIN {
        assert!(!g.push(&r(mid)), "オフに戻ったら ON までオフ");
    }
}
