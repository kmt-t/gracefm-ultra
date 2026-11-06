//! 受信（復号、MendNet、PostNet。`gfm_core::receiver`）。フレームを 1 つずつ受信機に入れる（実際の受信と同じ流れ）。使い方:
//!   gfm-rx --profile B200-L --frames a.sym --out out.raw [--models models/<profile>] [--ort libonnxruntime.so] [--threads 2]
//! a.sym（gfm-tx の出力）: u32 フレーム数, u32 フレームあたりのシンボル数, フレーム数個のフラグ (u8、1 = 届いた), 届いたフレームの I/Q (f32 リトルエンディアン、±1 = 1.0)。
//! out.raw: L/R の交互の f32（満点 ±1）、フレーム数 × ブロック長のサンプル。
//! --models: mendnet.onnx と postnet.onnx のディレクトリ（`tools/export_onnx.py` が書く）。
//! --ort: ONNX Runtime の共有ライブラリ（既定は環境変数 ORT_DYLIB_PATH）。--threads: 推論のスレッド数。
//! 標準エラー出力に、フレームの統計、超解像をかけた割合と切り替えの回数、処理時間（実時間比）、推論の処理時間（最初の 1 回と、残りの中央値・99%・最大）、
//! 遅延（確定待ち + 最初を除く最大の処理時間）、最大の常駐メモリを書く。

use gfm_core::post::{PostInput, PostModel, PostOutput, BINS, LOOKAHEAD, MDCT_HOP, REACH};
use gfm_core::profiles::profile_by_name;
use gfm_core::receiver::Receiver;
use gfm_core::restore::{MendBatch, MendModel, DELAY, NC, WIN};
use ort::session::Session;
use ort::value::Tensor;

fn read_sym(path: &str) -> Vec<Option<Vec<[f64; 2]>>> {
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
            let i = f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as f64;
            let q = f32::from_le_bytes([d[o + 4], d[o + 5], d[o + 6], d[o + 7]]) as f64;
            fr.push([i, q]);
            o += 8;
        }
        out.push(Some(fr));
    }
    out
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn tensor<T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static>(
    shape: &[usize],
    v: &[T],
) -> Result<Tensor<T>, String> {
    Tensor::from_array((shape.to_vec(), v.to_vec())).map_err(err)
}

/// mendnet.onnx の推論
struct OrtMend(Session);

impl MendModel for OrtMend {
    fn infer(&mut self, w: &MendBatch) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (b, nb, k) = (w.b, w.nb, w.nbands);
        let out = self
            .0
            .run(ort::inputs![
                "coef" => tensor(&[b, WIN, NC], &w.coef)?,
                "var" => tensor(&[b, WIN, k], &w.var)?,
                "status" => tensor(&[b, WIN], &w.status)?,
                "gap" => tensor(&[b, WIN], &w.gap)?,
                "comp" => tensor(&[b], &w.comp)?,
                "hint_h" => tensor(&[b, WIN, NC], &w.hint_h)?,
                "hint_q" => tensor(&[b, WIN, k], &w.hint_q)?,
                "llr" => tensor(&[b, WIN, nb], &w.llr)?,
                "dstep" => tensor(&[b, WIN, NC], &w.dstep)?,
            ])
            .map_err(err)?;
        let e0 = out["e0"].try_extract_tensor::<f32>().map_err(err)?.1.to_vec();
        let sd = out["sd"].try_extract_tensor::<f32>().map_err(err)?.1.to_vec();
        Ok((e0, sd))
    }
}

/// postnet.onnx の推論（1 歩）
struct OrtPost {
    session: Session,
    dims: Vec<(usize, usize)>,
}

impl OrtPost {
    fn new(session: Session) -> OrtPost {
        // 状態の入力 c0, c1, … の形 (2, チャンネル, 幅)
        let dims = session
            .inputs()
            .iter()
            .filter(|i| i.name().starts_with('c'))
            .map(|i| {
                let s = i.dtype().tensor_shape().expect("状態の入力はテンソル");
                (s[1] as usize, s[2] as usize)
            })
            .collect();
        OrtPost { session, dims }
    }
}

impl PostModel for OrtPost {
    fn state_dims(&self) -> Vec<(usize, usize)> {
        self.dims.clone()
    }

    fn step(&mut self, x: &PostInput) -> Result<PostOutput, String> {
        let (f, n) = (x.frames, MDCT_HOP * (x.frames + 1) + 2 * REACH);
        let mut inputs = ort::inputs![
            "e0" => tensor(&[2, n, NC], x.e0)?,
            "sd" => tensor(&[2, n, NC], x.sd)?,
            "on" => tensor(&[f - LOOKAHEAD], x.on)?,
            "yprev" => tensor(&[2, BINS, 1], x.yprev)?,
        ];
        for (i, (&(ch, w), c)) in self.dims.iter().zip(x.cache).enumerate() {
            inputs.push((format!("c{i}").into(), tensor(&[2, ch, w], c)?.into()));
        }
        let out = self.session.run(inputs).map_err(err)?;
        let get = |name: &str| -> Result<Vec<f32>, String> { Ok(out[name].try_extract_tensor::<f32>().map_err(err)?.1.to_vec()) };
        Ok(PostOutput {
            audio: get("audio")?,
            y: get("y")?,
            u: (0..self.dims.len()).map(|i| get(&format!("u{i}"))).collect::<Result<_, _>>()?,
        })
    }
}

/// 最大の常駐メモリ [MB]（/proc/self/status の VmHWM）
fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut profile, mut frames, mut out) = (String::new(), String::new(), String::new());
    let mut models = String::new();
    let mut ort_path = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    let mut threads = 2usize;
    let mut i = 1;
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_else(|| panic!("{} に値がありません", args[i]));
        match args[i].as_str() {
            "--profile" => profile = next,
            "--frames" => frames = next,
            "--out" => out = next,
            "--models" => models = next,
            "--ort" => ort_path = next,
            "--threads" => threads = next.parse().unwrap(),
            a => panic!("不明な引数: {}", a),
        }
        i += 2;
    }
    if ort_path.is_empty() {
        panic!("ONNX Runtime の共有ライブラリを --ort か ORT_DYLIB_PATH で指定してください");
    }
    let p = profile_by_name(&profile).unwrap_or_else(|| panic!("不明なプロファイル: {}", profile));
    if models.is_empty() {
        models = format!("models/{}", profile);
    }
    let env = ort::init_from(&ort_path)
        .unwrap_or_else(|e| panic!("{}: {}", ort_path, e))
        .build()
        .unwrap_or_else(|e| panic!("ONNX Runtime の初期化に失敗: {}", e));
    let session = |name: &str| -> Session {
        let path = format!("{}/{}", models, name);
        let fail = |e: &dyn std::fmt::Display| -> ! { panic!("{}: {}", path, e) };
        let b = Session::builder(&env).unwrap_or_else(|e| fail(&e));
        let b = b.with_intra_threads(threads).unwrap_or_else(|e| fail(&e));
        // 推論の合間（次のフレームを待つ間）にスレッドを回し続けない
        let mut b = b.with_config_entry("session.intra_op.allow_spinning", "0").unwrap_or_else(|e| fail(&e));
        b.commit_from_file(&path).unwrap_or_else(|e| fail(&e))
    };
    let mut rx =
        Receiver::new(&p, OrtMend(session("mendnet.onnx")), OrtPost::new(session("postnet.onnx"))).unwrap_or_else(|e| panic!("{}", e));
    let fr = read_sym(&frames);
    let mut pcm: Vec<[f64; 2]> = Vec::with_capacity(fr.len() * p.block);
    let t0 = std::time::Instant::now();
    let mut steps: Vec<f64> = Vec::new(); // 音声を出したフレームの処理時間（網の推論を含む）
    for f in &fr {
        let t = std::time::Instant::now();
        let out = rx.push(f.as_deref()).unwrap_or_else(|e| panic!("{}", e));
        if !out.is_empty() {
            steps.push(t.elapsed().as_secs_f64());
        }
        pcm.extend(out);
    }
    pcm.extend(rx.finish().unwrap_or_else(|e| panic!("{}", e)));
    let dt = t0.elapsed().as_secs_f64();
    let mut raw = Vec::with_capacity(pcm.len() * 8);
    for s in &pcm {
        raw.extend((s[0] as f32).to_le_bytes());
        raw.extend((s[1] as f32).to_le_bytes());
    }
    std::fs::write(&out, raw).expect("書き出しに失敗");
    let dur = pcm.len() as f64 / p.fs as f64;
    let frame = p.block as f64 / p.fs as f64;
    let st = rx.stats();
    // 最初の推論（初期化を含む）は分けて、残りの分布を出す
    let first = steps.first().copied().unwrap_or(0.0);
    let mut rest = steps.get(1..).unwrap_or(&[]).to_vec();
    rest.sort_by(f64::total_cmp);
    let at = |q: f64| rest.get(((rest.len().max(1) - 1) as f64 * q) as usize).copied().unwrap_or(0.0);
    let worst = at(1.0);
    eprintln!(
        "frames {} lost {} crc_fail {} high_crc_fail {} recovered {} post_on {:.3} switches {} time {:.2}s rtf {:.3} first {:.1}ms median {:.1}ms p99 {:.1}ms worst {:.1}ms latency {:.1}ms peak_rss {:.0}MB",
        st.frames,
        st.lost,
        st.crc_fail,
        st.high_crc_fail,
        st.recovered,
        rx.post_on as f64 / st.frames.max(1) as f64,
        rx.switches,
        dt,
        dt / dur,
        first * 1e3,
        at(0.5) * 1e3,
        at(0.99) * 1e3,
        worst * 1e3,
        ((DELAY + REACH) as f64 * frame + worst) * 1e3,
        peak_rss_mb()
    );
}
