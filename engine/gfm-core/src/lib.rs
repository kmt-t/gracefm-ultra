//! GraceFM の送信側（符号化・フレーム化・変調）と、学習データ生成用の通信路。仕様は ../gracefm の Python 実装に合わせる。
pub mod codec;
pub mod dsp;
pub mod fec;
pub mod framing;
pub mod gen;
pub mod layout;
pub mod modem;
pub mod npz;
pub mod profiles;
