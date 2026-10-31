# 送受信の対称な設計

GraceFM のビット列・フレーム形式は変えない。Rust の実装（`engine/gfm-core`）は、送信と受信が鏡になる構成にする。

- **送信（`Encoder`）は整数**: ビット列の定義そのもの（FPGA の送信機と同じ）。本家 Python とビット一致する（`tools/test_rs_parity.py`）。
- **受信（`Decoder`、`receiver`）は浮動小数点（f64）**: Ultra の本番経路。DSP では動かさず、音質を優先する。本家の整数の受信とビット一致させない。
  本家の演算幅の仕様（乗算器 25×18 bit など）や、その修正は受信に持ち込まない。

## 処理の対

| 段 | 送信（整数） | 受信（浮動小数点） | 入出力 |
|---|---|---|---|
| 変換 | `analyze` | `synthesize` | PCM ブロック ⇄ 帯域係数 `[成分 M/S][帯域 k][係数 j]`（M/S 分離と Haar をまとめる） |
| 量子化 | `quantize` | `dequantize` | 帯域係数 ⇄ `BlockData`（仮数・指数・アッテネータ）。受信は軟判定で、係数と誤差分散を返す |
| フレーム化 | `pack` | `unpack` | `BlockData` ⇄ 送信ビット列（ヘッダ、FEC、CRC、配置は `Layout`）。どちらも純粋な関数。Chase 救済は受信側の方針（`unpack_recovering`） |
| スクランブル | `scramble` | `descramble` | ビット ⇄ （LLR, 硬判定）。同じ PRBS |
| 変調 | `modulate` | `demodulate` | ビット ⇄ シンボル（受信は LLR・硬判定・雑音分散） |
| フレーム | `Encoder::encode_frame` | `Decoder::decode_frame` | PCM ブロック ⇄ シンボル（受信は欠落を `None` で受ける） |
| ストリーム | `Encoder::run` | `Decoder::run` | 全体の時刻合わせ（遅延 `offset`）を両側で同じ契約にする |

受信にだけある段（送信に対応なし）: 2 系統の融合（逆分散重み）、欠落補間（PLC、LPC 外挿）、クロスフェード、出力 IIR 切替（真の Butterworth、4 次、4 kHz）。`receiver` に分け、フレームの復号とは別の「出力段」とする。

## 型

- `BlockData`: 量子化後の 1 ブロック（`quantize` の出力、`pack` の入力、`unpack` の出力、`dequantize` の入力）。
- `Unpacked`: `unpack` の出力。`BlockData`、`high_ok`、FEC で訂正した保護ビット（`protected_bits`）。
- `DecodedFrame`: `decode_frame` の出力。PCM（満点 ±1）、有効フラグ、品質コード、誤差分散 `rb`。副作用で持たず、戻り値で返す。
- `Estimate`: `dequantize` の出力。係数と、半ブロック・成分・帯域ごとの誤差分散（線形、情報なしは `INFINITY`）。

## 受信の浮動小数点の中身

- LLR は自然対数の最大対数近似 `(D1 - D0) / (2σ²)`、σ² は最寄りの信号点との差から。ビタビの経路メトリックも浮動小数点。
- 仮数の各ビットが 1 である確率 p1 は厳密な sigmoid。FEC で訂正した保護ビットは確定値（0 か 1）。係数は期待値、誤差分散は `Σ p1(1-p1)·4^pos + 1/12`（刻みの二乗倍）。
- 逆 Haar は直交（1/√2）。L = M + S、R = M - S。
- 2 系統の融合は、帯域ごと・半ブロックごとに w_A = R_B / (R_A + R_B)。情報なしは 0 / 1 / 1/2。

## 本家（Python）の非対称（本家のリファクタで解消済み）

1. スクランブルが送受信に直書き → `scramble` / `descramble`。
2. `encode_block` は一括、受信は分散して逆量子化が 2 か所に重複 → `analyze`/`synthesize`、`quantize`/`dequantize`。
3. `unpack` が `header` 辞書で訂正済みビットを漏らす → `Unpacked`。Chase 救済は外へ。
4. `StreamDecoder` の隠れた状態 → `decode_frame` の戻り値。
5. `receive()` が融合・PLC・IIR を一括 → 出力段として分離。

## Rust のモジュール

```
codec/transform.rs   analyze（整数）/ synthesize（浮動小数点）、Haar
codec/quant.rs       quantize（整数）/ dequantize（浮動小数点）、BlockData、Estimate
framing.rs           pack / unpack / unpack_recovering
fec.rs               rsc_parity、CRC（送信）、ビタビ（受信）
scrambler.rs         scramble / descramble
modem.rs             modulate / demodulate
encoder.rs           Encoder
decoder.rs           Decoder
fusion.rs plc.rs iir.rs receiver.rs   出力段
gen.rs               学習データ生成（Encoder と通信路）
```
