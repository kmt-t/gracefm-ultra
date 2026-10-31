//! 音声符号化。送信 `encode_block` = analyze → quantize（整数）、受信 = dequantize → synthesize（浮動小数点）。
pub mod quant;
pub mod transform;

pub use quant::*;
pub use transform::*;

use crate::profiles::Profile;

/// 送信: PCM ブロック（符号つき 24 bit の L/R）→ BlockData
pub fn encode_block(pcm: &[[i64; 2]], p: &Profile, clip_opt: bool) -> BlockData {
    quantize_block(&analyze(pcm, p), p, clip_opt)
}
