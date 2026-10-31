//! 音声符号化。送信 `encode_block` = analyze → quantize。受信は decoder.rs（soft_coeffs → synthesize）。浮動小数点。
pub mod quant;
pub mod transform;

pub use quant::*;
pub use transform::*;

use crate::profiles::Profile;

/// 送信: PCM ブロック（満点 ±1 の L/R）→ BlockData。analyze → quantize
pub fn encode_block(pcm: &[[f64; 2]], p: &Profile, clip_opt: bool) -> BlockData {
    quantize(&analyze(pcm, p), p, clip_opt)
}
