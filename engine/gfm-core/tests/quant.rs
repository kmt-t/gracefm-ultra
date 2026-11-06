//! 量子化（quantize_band / quantize）と、変換（analyze_stream / synthesize_stream）。

mod common;
use common::{stereo, Rng, PROFILES};
use gfm_core::codec::*;
use gfm_core::profiles::profile_by_name;

#[test]
fn analysis_and_synthesis_are_inverse() {
    for name in ["B200", "B200-L"] {
        let p = profile_by_name(name).unwrap();
        let x = stereo(64 * 8, 1);
        let y = synthesize_stream(&analyze_stream(&x, &p), &p);
        for (a, b) in x.iter().zip(&y) {
            assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12, "{}", name);
        }
    }
}

#[test]
fn a_haar_frame_stays_in_its_block() {
    let p = profile_by_name("B200").unwrap();
    let mut frames = analyze_stream(&vec![[0.0, 0.0]; 64 * 5], &p);
    for band in &mut frames[2][0] {
        band.fill(1.0);
    }
    let y = synthesize_stream(&frames, &p);
    let on: Vec<usize> = (0..y.len()).filter(|&i| y[i][0] != 0.0).collect();
    assert!(on[0] >= 2 * 64 && *on.last().unwrap() < 3 * 64);
}

#[test]
fn the_haar_lifting_follows_gracefm() {
    // d = o − e、a = e + d / 2。帯域の並びは [低域, 詳細（低→高）]
    let bands = haar_forward(&[1.0, 3.0, 2.0, 6.0], 2);
    assert_eq!(bands, vec![vec![3.0], vec![2.0], vec![2.0, 4.0]]);
    assert_eq!(haar_inverse(&bands), vec![1.0, 3.0, 2.0, 6.0]);
}

#[test]
fn a_frame_reaches_15_samples_before_and_16_after() {
    let p = profile_by_name("B200-L").unwrap();
    let mut frames = analyze_stream(&vec![[0.0, 0.0]; 64 * 5], &p);
    for band in &mut frames[2][0] {
        band.fill(1.0);
    }
    let y = synthesize_stream(&frames, &p);
    let on: Vec<usize> = (0..y.len()).filter(|&i| y[i][0] != 0.0).collect();
    assert_eq!((on[0], *on.last().unwrap()), (2 * 64 - 15, 3 * 64 + 15));
}

#[test]
fn coef_power_is_the_energy_of_the_basis() {
    for name in ["B200", "B200-L"] {
        let p = profile_by_name(name).unwrap();
        let power = coef_power(&p);
        let mut o = 0;
        for k in 0..p.nbands() {
            let mut frames = analyze_stream(&vec![[0.0, 0.0]; 64 * 8], &p);
            frames[4][0][k][0] = 1.0;
            let e: f64 = synthesize_stream(&frames, &p).iter().map(|v| v[0] * v[0]).sum();
            assert!((e - power[o]).abs() < 1e-12, "{} 帯域 {k}", name);
            o += p.ncoef(k);
        }
    }
}

#[test]
fn quantization_error_is_within_half_a_step() {
    let mut rng = Rng::new(2);
    for name in PROFILES {
        let p = profile_by_name(name).unwrap();
        for ch in 0..2 {
            for k in 0..p.nbands() {
                let b = p.bits(ch)[k];
                if b == 0 {
                    continue;
                }
                let n = p.ncoef(k);
                let c: Vec<f64> = (0..n).map(|_| (rng.uniform() - 0.5) * 0.6).collect();
                let (m, eidx, attn) = quantize_band(&c, ch, k, &p, false);
                let (emin, emax) = exp_range(b, &p);
                assert!((0..=emax - emin).contains(&eidx), "{} ch{} k{}: 指数 {}", name, ch, k, eidx);
                let e = eidx + emin;
                for j in 0..n {
                    let step = 2f64.powi((e - p.attn_at(ch, k, &attn, j)) as i32);
                    assert!(m[j].abs() <= qmax_of(b), "仮数が範囲外");
                    // 飽和していなければ、誤差は刻みの半分以内
                    if (m[j].abs()) < qmax_of(b) {
                        assert!((c[j] - m[j] as f64 * step).abs() <= step / 2.0 + 1e-15, "{} ch{} k{} j{}", name, ch, k, j);
                    }
                }
            }
        }
    }
}

#[test]
fn silence_and_overload_are_handled() {
    let p = profile_by_name("B200-L").unwrap();
    let (m, eidx, _) = quantize_band(&[0.0; 4], 0, 0, &p, true);
    assert!(m.iter().all(|&v| v == 0) && eidx == 0, "無音は仮数 0、指数は最小");
    let b = p.bits(0)[0];
    let (m, _, _) = quantize_band(&[1e9, -1e9, 0.0, 0.0], 0, 0, &p, true);
    assert_eq!((m[0], m[1]), (qmax_of(b), qmin_of(b)), "大きすぎる値は飽和する");
}

#[test]
fn attenuator_never_increases_the_error() {
    // B200 の M 帯域 k=2 は 2 係数ごとに 1 bit のアッテネータ。アッテネータで刻みを半分にして誤差が減る区間だけ 1 になる
    let p = profile_by_name("B200-L").unwrap();
    let mut rng = Rng::new(3);
    let c: Vec<f64> = (0..8).map(|_| (rng.uniform() - 0.5) * 0.2).collect();
    let (m, eidx, attn) = quantize_band(&c, 0, 2, &p, false);
    assert_eq!(attn.len(), 4);
    let (emin, _) = exp_range(p.bits(0)[2], &p);
    let step = 2f64.powi((eidx + emin) as i32);
    for (g, &used_attn) in attn.iter().enumerate() {
        let err = |half: bool, mm: &[i64]| -> f64 {
            (2 * g..2 * g + 2).map(|j| (c[j] - mm[j] as f64 * if half { step / 2.0 } else { step }).powi(2)).sum()
        };
        let used = err(used_attn == 1, &m);
        let m0: Vec<i64> = c.iter().map(|v| (v / step + 0.5).floor() as i64).collect();
        assert!(used <= err(false, &m0) + 1e-18, "区間 {} でアッテネータが誤差を増やした", g);
    }
}

#[test]
fn clip_optimization_never_increases_the_error() {
    let p = profile_by_name("B200-L").unwrap();
    let mut rng = Rng::new(4);
    // まれに大きなピークがある信号: 指数を 1 つ下げて、ピークを飽和させるほうが誤差が小さくなりうる
    let mut c: Vec<f64> = (0..32).map(|_| (rng.uniform() - 0.5) * 0.01).collect();
    c[3] = 0.5;
    let recon = |clip: bool| -> f64 {
        let (m, eidx, attn) = quantize_band(&c, 0, 4, &p, clip);
        let (emin, _) = exp_range(p.bits(0)[4], &p);
        (0..32).map(|j| (c[j] - m[j] as f64 * 2f64.powi((eidx + emin - p.attn_at(0, 4, &attn, j)) as i32)).powi(2)).sum()
    };
    assert!(recon(true) <= recon(false) + 1e-18);
}
