# 送受信の対称な設計

GraceFM のビット列・フレーム形式は変えない。Rust の実装（`engine/gfm-core`）は、送信と受信が鏡になる構成にする。
Python 版（`../gracefm`）へのバックポート用に、対応づけと、現状の非対称をまとめる。

**状況（2026-11-01）**: 本家の Python 版は、この構成に沿ってリファクタされた（本家のコミット 60602fd。`encoder.py` / `decoder.py` / `receiver.py` / `scrambler.py`、旧名の別名なし）。Rust 版は Python 版に追従し、2 つの一致テスト（`tools/test_rs_parity.py`、`tools/test_rs_rx_parity.py`）で完全一致を確認している。本家は続けて出力 IIR の係数を 18 bit（Q16）にした（作業ツリーの変更）。Rust の `iir.rs` と定数表（`tools/gen_rs_tables.py` が `tables.py` から作る）は、その版に合わせてある。

## 処理の対

| 段 | 送信 | 受信 | 入出力 |
|---|---|---|---|
| 変換 | `analyze` | `synthesize` | PCM ブロック ⇄ 帯域係数 `[成分 M/S][帯域 k][係数 j]`（M/S 分離と Haar をまとめる） |
| 量子化 | `quantize` | `dequantize` | 帯域係数 ⇄ `BlockData`（仮数・指数・アッテネータ）。受信は軟判定で、係数と誤差分散を返す |
| フレーム化 | `pack` | `unpack` | `BlockData` ⇄ 送信ビット列（ヘッダ、FEC、CRC、配置は `Layout`）。どちらも純粋な関数 |
| スクランブル | `scramble` | `descramble` | ビット ⇄ （LLR, 硬判定）。同じ PRBS |
| 変調 | `modulate` | `demodulate` | ビット ⇄ シンボル（受信は LLR・硬判定・雑音分散） |
| フレーム | `Encoder::encode_frame` | `Decoder::decode_frame` | PCM ブロック ⇄ シンボル（受信は欠落を `None` で受ける） |
| ストリーム | `Encoder::run` | `Decoder::run` | 全体の時刻合わせ（遅延 `offset`）を両側で同じ契約にする |

受信にだけある段（送信に対応なし）: 2 系統の融合、欠落補間（PLC）、クロスフェード、出力 IIR 切替。`receiver` に分け、フレームの復号とは別の「出力段」とする。

## 型

- `BlockData`: 量子化後の 1 ブロック（`quantize` の出力、`pack` の入力、`unpack` の出力、`dequantize` の入力）。
- `Unpacked`: `unpack` の出力。`BlockData`、`high_ok`、FEC で訂正した保護ビット（`protected_bits`）。受信側が事後確率を確定値にするのに使う。
- `DecodedFrame`: `decode_frame` の出力。PCM、有効フラグ、品質コード、誤差分散 `rb`。**副作用で持たず、戻り値で返す。**
- `Estimate`: `dequantize` の出力。係数と、半ブロック・成分・帯域ごとの誤差分散（対数領域、情報なしは `INF_LOG`）。

## 硬判定は軟判定の特殊ケース

`dequantize` は、仮数の各ビットが 1 である確率 p1（Q15）を受ける。p1 が 0 か 32768 の確定値のとき、期待値は硬判定の復号と一致し、分散は量子化誤差だけになる。
従来の `codec.decode_block`（硬判定）と `StreamDecoder.soft_coeffs`（軟判定）の二重実装は、1 つにする。

## 現状（Python）の非対称

1. スクランブルが送信 `frame()` と受信 `decode()` に直書き。送信はビット、受信は LLR と硬判定で、逆の関係が名前になっていない。
2. 送信 `encode_block` は「M/S 分離＋Haar＋量子化」を一括。受信は `soft_coeffs`（デコーダ内）、`haar_inverse`、`ms_to_lr` に分散し、`codec.decode_block` の硬判定経路も別にあり、逆量子化が 2 か所に重複する。
3. `unpack` が FEC 復号、CRC、解析、仮数組み立てを持ち、`header` 辞書で訂正済みビットを外へ漏らす。Chase 救済は受信側の方針なので `unpack` の外へ出す。
4. `StreamDecoder` が `last_quality`、`last_rb` などの隠れた状態を持つ。戻り値にする。
5. `receive()` が融合、PLC、クロスフェード、IIR 切替を一括で行う。出力段として分ける。

## Rust のモジュール

```
codec/transform.rs   analyze / synthesize（haar_forward / haar_inverse、M/S）
codec/quant.rs       quantize / dequantize（BlockData、Estimate）
codec/mod.rs         encode_block / decode_block
framing.rs           pack / unpack（純粋）
fec.rs               rsc_parity / rsc_decode、CRC
scrambler.rs         scramble / descramble
modem.rs             modulate / demodulate
encoder.rs           Encoder
decoder.rs           Decoder（Chase 救済を含む）
receiver.rs          出力段: 融合、クロスフェード、PLC、IIR 切替
fusion.rs plc.rs iir.rs
gen.rs               学習データ生成（Encoder と通信路）
```
