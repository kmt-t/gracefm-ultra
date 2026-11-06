//! 音声符号化。送信は analyze_stream → quantize。受信は decoder.rs（soft_coeffs）。浮動小数点。
pub mod quant;
pub mod transform;

pub use quant::*;
pub use transform::*;
