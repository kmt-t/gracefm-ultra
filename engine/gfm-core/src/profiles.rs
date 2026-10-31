//! プロファイル定義（GraceFM `gracefm/profiles.py` の移植。Y 方式と暗号化は未対応）。

/// 帯域数の上限（低域近似 1 + 詳細帯域 levels。登録プロファイルはすべて 5）
pub const MAX_BANDS: usize = 5;

#[derive(Clone, Debug)]
pub struct Profile {
    pub pid: u32,
    pub name: &'static str,
    pub fs: u32,
    pub block: usize,
    pub levels: usize,
    pub bits_m: [i64; MAX_BANDS],
    pub bits_s: [i64; MAX_BANDS],
    pub exp_bits: usize,
    pub protect: [usize; MAX_BANDS],
    pub qam16h: bool,
    pub alpha: i64,
    pub base_m: Option<[i64; MAX_BANDS]>,
    pub base_s: Option<[i64; MAX_BANDS]>,
    pub attn_group_m: [usize; MAX_BANDS],
    pub attn_group_s: [usize; MAX_BANDS],
    pub y_start_k: Option<usize>,
    pub format_id: u32,
}

fn levels_for(fs: u32, block: usize) -> usize {
    let mut levels = 1;
    while fs >= (4000u32 << levels) {
        levels += 1;
    }
    assert!(block >> levels >= 1);
    levels
}

pub fn band_level(k: usize, levels: usize) -> usize {
    if k == 0 {
        levels
    } else {
        levels + 1 - k
    }
}

impl Profile {
    pub fn nbands(&self) -> usize {
        self.levels + 1
    }
    pub fn ncoef(&self, k: usize) -> usize {
        self.block >> band_level(k, self.levels)
    }
    /// ch: 0 = M, 1 = S
    pub fn bits(&self, ch: usize) -> &[i64; MAX_BANDS] {
        if ch == 0 {
            &self.bits_m
        } else {
            &self.bits_s
        }
    }
    pub fn attenuator_group(&self, ch: usize, k: usize) -> usize {
        if ch == 0 {
            self.attn_group_m[k]
        } else {
            self.attn_group_s[k]
        }
    }
    /// 帯域 (ch, k) のアッテネータのビット数（区間の数）。使わない帯域は 0
    pub fn attenuator_bits(&self, ch: usize, k: usize) -> usize {
        self.ncoef(k).checked_div(self.attenuator_group(ch, k)).unwrap_or(0)
    }
    /// 係数 j のアッテネータ（0 か 1。刻みを半分にするか）。attn はその帯域のアッテネータ。区間を使わない帯域は 0
    pub fn attn_at(&self, ch: usize, k: usize, attn: &[i64], j: usize) -> i64 {
        j.checked_div(self.attenuator_group(ch, k)).map_or(0, |i| attn[i])
    }
    pub fn layered(&self) -> bool {
        self.base_m.is_some()
    }
    pub fn base_bits(&self, ch: usize, k: usize) -> i64 {
        if !self.layered() {
            return self.bits(ch)[k];
        }
        if ch == 0 {
            self.base_m.unwrap()[k]
        } else {
            self.base_s.unwrap()[k]
        }
    }
    pub fn protect_bits(&self, k: usize) -> usize {
        self.protect[k]
    }
    pub fn high_k(&self) -> usize {
        self.nbands() - 2
    }
    pub fn frames_per_sec(&self) -> u32 {
        self.fs / self.block as u32
    }
}

#[allow(clippy::too_many_arguments)]
fn row(
    pid: u32,
    name: &'static str,
    fs: u32,
    bits_m: [i64; MAX_BANDS],
    bits_s: [i64; MAX_BANDS],
    protect: [usize; MAX_BANDS],
    qam16h: bool,
    base_m: Option<[i64; MAX_BANDS]>,
    base_s: Option<[i64; MAX_BANDS]>,
    attn_m: [usize; MAX_BANDS],
    attn_s: [usize; MAX_BANDS],
    y: Option<usize>,
) -> Profile {
    Profile {
        pid,
        name,
        fs,
        block: 64,
        levels: levels_for(fs, 64),
        bits_m,
        bits_s,
        exp_bits: 4,
        protect,
        qam16h,
        alpha: if qam16h { 5 } else { 1 },
        base_m,
        base_s,
        attn_group_m: attn_m,
        attn_group_s: attn_s,
        y_start_k: y,
        format_id: 3,
    }
}

pub fn profile_by_name(name: &str) -> Option<Profile> {
    let z = [0i64; MAX_BANDS];
    let z5 = [0usize; MAX_BANDS];
    let p2 = [2usize, 1, 1, 1, 1];
    let q1 = [1usize; MAX_BANDS];
    Some(match name {
        "B200M" => row(1, "B200M", 32000, [10, 11, 7, 7, 7], z, p2, true, Some([4, 5, 3, 3, 3]), Some(z), [0, 0, 2, 0, 4], z5, None),
        "B200" => row(
            2,
            "B200",
            32000,
            [10, 11, 7, 7, 7],
            [10, 7, 7, 7, 7],
            p2,
            true,
            Some([4, 5, 3, 3, 3]),
            Some([4, 3, 3, 3, 3]),
            [0, 0, 2, 0, 4],
            [0, 0, 2, 0, 4],
            None,
        ),
        "B200MY" => row(3, "B200MY", 32000, [10, 11, 7, 7, 7], z, p2, true, Some([4, 5, 1, 1, 1]), Some(z), [0, 0, 2, 4, 8], z5, Some(2)),
        "B200Y" => row(
            4,
            "B200Y",
            32000,
            [10, 11, 7, 7, 7],
            [10, 11, 7, 7, 7],
            p2,
            true,
            Some([4, 5, 1, 1, 1]),
            Some([4, 5, 1, 1, 1]),
            [0, 0, 2, 4, 8],
            [0, 0, 2, 4, 8],
            Some(2),
        ),
        "SUB50M" => row(5, "SUB50M", 32000, [7, 5, 3, 0, 0], z, q1, false, None, None, [0, 0, 4, 0, 0], z5, None),
        "SUB100" => {
            row(6, "SUB100", 32000, [7, 7, 3, 0, 0], [5, 3, 3, 0, 0], q1, false, None, None, [0, 0, 4, 0, 0], [0, 0, 4, 0, 0], None)
        }
        "M700M" => row(7, "M700M", 48000, [10, 11, 7, 7, 7], z, p2, true, Some([4, 5, 3, 3, 3]), Some(z), [0, 0, 2, 0, 4], z5, None),
        "M700" => row(
            8,
            "M700",
            48000,
            [10, 11, 7, 7, 7],
            [10, 7, 7, 7, 3],
            p2,
            true,
            Some([4, 5, 3, 3, 3]),
            Some([4, 3, 3, 3, 1]),
            [0, 0, 2, 0, 4],
            [0, 0, 2, 0, 4],
            None,
        ),
        "M700MY" => row(9, "M700MY", 48000, [10, 11, 7, 7, 7], z, p2, true, Some([4, 5, 1, 1, 1]), Some(z), [0, 0, 2, 4, 8], z5, Some(2)),
        "M700Y" => row(
            10,
            "M700Y",
            48000,
            [10, 11, 7, 7, 7],
            [10, 11, 7, 7, 7],
            p2,
            true,
            Some([4, 5, 1, 1, 1]),
            Some([4, 5, 1, 1, 1]),
            [0, 0, 2, 4, 8],
            [0, 0, 2, 4, 8],
            Some(2),
        ),
        "SUB100-48K" => {
            row(11, "SUB100-48K", 48000, [7, 7, 3, 0, 0], [5, 3, 0, 0, 0], q1, false, None, None, [0, 0, 4, 0, 0], [0, 4, 0, 0, 0], None)
        }
        _ => return None,
    })
}
