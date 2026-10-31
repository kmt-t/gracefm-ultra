//! 音声符号化。送信 `encode_block` = analyze → quantize、受信 `decode_block` = dequantize → synthesize。
//! すべて整数演算（GraceFM `gracefm/codec.py` の移植）。
pub mod quant;
pub mod transform;

pub use quant::*;
pub use transform::*;

use crate::profiles::Profile;

/// 送信: PCM ブロック（符号つき 24 bit の L/R）→ BlockData
pub fn encode_block(pcm: &[[i64; 2]], p: &Profile, clip_opt: bool) -> BlockData {
    quantize_block(&analyze(pcm, p), p, clip_opt)
}

/// 受信（硬判定）: BlockData → PCM ブロック。通信路なしの往復や試験に使う
pub fn decode_block(bd: &BlockData, p: &Profile) -> Vec<[i64; 2]> {
    synthesize(&dequantize_hard(bd, p))
}
