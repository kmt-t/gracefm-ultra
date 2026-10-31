//! 欠落補間（PLC）。LPC 外挿（倍精度）。GraceFM `gracefm/plc.py` に相当。

pub struct PlcParams {
    pub order: usize,
    pub hist_len: usize,
    pub hold_ms: f64,
    pub fade_ms: f64,
    pub xfade: usize,
}

impl Default for PlcParams {
    fn default() -> Self {
        PlcParams { order: 16, hist_len: 128, hold_ms: 10.0, fade_ms: 20.0, xfade: 16 }
    }
}

/// ハン窓をかけた履歴の自己相関から Levinson-Durbin で LPC 係数 a[0..=order]（a[0] = 1）を求める。白色雑音補正と帯域拡張（0.998^i）つき
pub fn lpc(x: &[f64], order: usize) -> Vec<f64> {
    let n = x.len();
    let denom = (n.max(2) - 1) as f64;
    let w: Vec<f64> = x.iter().enumerate().map(|(i, &v)| v * 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos())).collect();
    let mut r: Vec<f64> = (0..=order).map(|k| (0..n.saturating_sub(k)).map(|i| w[i] * w[i + k]).sum()).collect();
    let mut a = vec![0f64; order + 1];
    a[0] = 1.0;
    if r[0] <= 0.0 {
        return a;
    }
    r[0] *= 1.0 + 1e-4;
    let mut err = r[0];
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = (-acc / err).clamp(-0.99997, 0.99997);
        let old = a.clone();
        for j in 1..i {
            a[j] = old[j] + k * old[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            break;
        }
    }
    let mut bw = 1.0;
    for i in 1..=order {
        bw *= 0.998;
        a[i] *= bw;
    }
    a
}

pub fn extrapolate(hist: &[f64], n: usize, order: usize) -> Vec<f64> {
    if hist.len() <= order {
        return vec![0.0; n];
    }
    let a = lpc(hist, order);
    let mut buf: Vec<f64> = hist[hist.len() - order..].to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut acc = 0.0;
        for j in 0..order {
            acc += a[1 + j] * buf[buf.len() - 1 - j];
        }
        let v = (-acc).clamp(-1.0, 1.0);
        out.push(v);
        buf.push(v);
        buf.remove(0);
    }
    out
}

/// 欠落区間を LPC 外挿で埋める。hold_ms までは等倍、その後 fade_ms で 0 へ。復帰時は xfade サンプルで外挿値から受信値へクロスフェード
pub fn conceal(y: &[[f64; 2]], valid: &[bool], fs: usize, pp: &PlcParams) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = y.iter().zip(valid).map(|(s, &v)| if v { *s } else { [0.0, 0.0] }).collect();
    let hold = fs as f64 * pp.hold_ms / 1000.0;
    let fade = (fs as f64 * pp.fade_ms / 1000.0).max(1.0);
    let n = y.len();
    let mut i = 0;
    while i < n {
        if valid[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !valid[j] {
            j += 1;
        }
        let l = j - i;
        let k = pp.xfade.min(n - j);
        for c in 0..2 {
            let hist: Vec<f64> = out[i.saturating_sub(pp.hist_len)..i].iter().map(|v| v[c]).collect();
            let ext = extrapolate(&hist, l + k, pp.order);
            let pred: Vec<f64> = ext.iter().enumerate().map(|(t, &e)| e * (1.0 - (t as f64 - hold) / fade).clamp(0.0, 1.0)).collect();
            for t in 0..l {
                out[i + t][c] = pred[t];
            }
            for m in 0..k {
                let w = (m as f64 + 1.0) / (k as f64 + 1.0);
                out[j + m][c] = w * out[j + m][c] + (1.0 - w) * pred[l + m];
            }
        }
        i = j;
    }
    out
}

pub fn crossfade_switch(y: &[[f64; 2]], patterns: &[Vec<bool>], block: usize, pp: &PlcParams) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = y.to_vec();
    let nb = patterns.len().min(y.len() / block);
    for i in 1..nb {
        if patterns[i] != patterns[i - 1] && patterns[i].iter().any(|&b| b) && patterns[i - 1].iter().any(|&b| b) {
            let s = i * block;
            let k = pp.xfade.min(y.len() - s);
            for c in 0..2 {
                let hist: Vec<f64> = out[s.saturating_sub(pp.hist_len)..s].iter().map(|v| v[c]).collect();
                let pred = extrapolate(&hist, k, pp.order);
                for m in 0..k {
                    let w = (m as f64 + 1.0) / (k as f64 + 1.0);
                    out[s + m][c] = w * out[s + m][c] + (1.0 - w) * pred[m];
                }
            }
        }
    }
    out
}

