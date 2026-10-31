//! 生の PCM ファイルの読み込み: 符号つき 24 bit を int32（リトルエンディアン）で入れた L/R の交互（満点 = 2^23）。

/// 生データ → 満点 ±1 の L/R。端の半端なバイトは捨てる
pub fn read_pcm24_le(raw: &[u8]) -> Vec<[f64; 2]> {
    let full_scale = (1u64 << 23) as f64;
    let (frames, _) = raw.as_chunks::<8>();
    frames
        .iter()
        .map(|c| {
            let l = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let r = i32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            [l as f64 / full_scale, r as f64 / full_scale]
        })
        .collect()
}
