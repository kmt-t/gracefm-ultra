# 受信機の設計（Rust、`engine/gfm-core`）

GraceFM のビット列・フレーム形式は変えず、送信側は本家 Python（`../gracefm`）と対になる構成、受信側は復号のあとに MendNet と PostNet を置く構成とする。

## 基本方針

| 項目 | 方針 |
|---|---|
| **帯域分割** | 本家と同じく Haar（S 変換、`format_id 3`）と LeGall 5/3（`-L`、`format_id 2`）をプロファイルで選ぶ。網はプロファイルごとに学習する。本家の出力段（PLC・IIR 切替・再標本化）は使わず、PostNet の出力をプロファイルの標本化周波数のまま出す。 |
| **音声処理** | 符号化と受信は浮動小数点で行う。DSP 向けの固定小数点仕様は持ち込まないため、音声サンプルは本家とビット単位では一致しない（音質優先）。 |
| **ビット処理** | ビット配置・FEC・CRC・スクランブルは本家とビット単位で同一。Rust 送信 → 本家 Python 受信の試験（`tools/test_rs_compat.py`）で検証する。 |
| **網の実行** | 入力特徴の構築と出力の後処理も ONNX グラフに含める（`tools/export_onnx.py`）。Rust 側は観測と状態を並べて渡すだけにする。 |

## 受信の流れ

フレームを 1 つずつ投入する。各段は次段に必要なデータだけを受け渡す。

```
シンボル ─▶ Decoder ─DecodedFrame─▶ Observation ─▶ Mender ─Mended─▶ Gate ─(Restored, 判定)─▶ PostFilter ─▶ 音声 L/R
            復号                     観測の符号化   MendNet          判定                     PostNet
```

| 段 | 型 | 役割 | 出力 |
|---|---|---|---|
| **復号** | `decoder::Decoder` | 復調、スクランブル解除、ヘッダ復号（事前知識つき → 失敗時はリスト復号）、FEC、軟判定の逆量子化。直前までのヘッダを記憶する。 | `DecodedFrame`: 状態、帯域係数、帯域別誤差分散、仮数ビットの受信 LLR、ヘッダ、`gap`、壊れたフレームの手がかり |
| **観測** | `restore::Observation` | MendNet 入力仕様どおりに符号化（LLR 量子化、刻み、情報なしは `−1`）。 | 1 フレームの観測 |
| **補間** | `restore::Mender` | 直近 100 フレームの窓を 50 フレームごとに MendNet へかけ、中央 50 フレームを確定する（最初の窓のみ先頭から）。 | `Mended`: 確定フレーム（`e0`, `sd`, `usable`）と窓残りの暫定フレーム |
| **判定** | `gate::Gate` | 過去 500 フレーム（32 kHz で 1 秒）の使えたフレームから予測 SNR を算出。27.5 dB 以上でオン、オン中は 25.5 dB 未満になるまで維持。 | フレームごとの判定（`bool`） |
| **後段** | `post::PostFilter` | PostNet を段ごとの状態を持ち越して逐次実行する。確定フレームだけで決まる位置まで状態を進め、暫定フレームに依存する先読み部分は毎回再計算する。 | 音声（L/R、満点 ±1） |
| **受信** | `receiver::Receiver` | 上記をフレーム単位で接続し、超解像の適用割合と切り替え回数を記録する。 | 音声 |

網の推論はトレイト `restore::MendModel` と `post::PostModel` に抽象化している。実運用は `gfm-rx` が ONNX Runtime で実装し、単体試験（`tests/stream.rs`）では入力をそのまま返すダミーモデルで窓と区間の受け渡しを検証する。

### 遅延

全体の遅延は **152 ms（32 kHz の場合。48 kHz は約 101 ms）+ 1 回分の計算時間** で、`gfm-rx` が実測値 `latency` を出力する。

| 要素 | 内容 | 遅延（32 kHz 時） |
|---|---|---|
| 補間の確定待ち | 窓 `[a, a + 100)` は `a + 99` 到達時に `[a + 25, a + 75)` を確定。最長待ちは 75 フレーム。 | 150 ms |
| 合成の届く範囲 | フレーム末尾の標本には次フレームの係数が届くため、確定フレームより 1 フレーム手前まで出力（確定境界の継ぎ目防止）。 | 2 ms |
| 後段の先読み | MDCT 2 フレーム + 窓重なり + 合成の隣 1 フレーム（最大 16 フレーム）は、補間窓の残り 25 フレーム（暫定値）で賄う。 | 0 ms（増加なし） |
| 送信側の先読み | `-L`（LeGall 5/3）は符号化に 15 サンプルの先読みが必要。Haar は 0。 | 0.47 ms（送信側） |
| 推論計算 | 50 フレームごとに MendNet 1 回（窓 1 つ × M/S）と PostNet 1 歩を実行。 | 実測 12〜20 ms（最大） |

- **実測値**（`B200` / `B200-L`、30 秒、Es/N0 24 dB、欠落 5%、推論 2 スレッド、専用 CPU 2 コア、各 4 回）:
  - 計算時間: 中央値 10 ms、最大 12〜20 ms
  - 総遅延: **164〜172 ms**、実時間比 `0.14`、メモリ `169 MB`（※ファイル入出力ベースであり、RTL-SDR の I/Q 同期・等化と音声出力バッファは含まない）。
- **注意**: CPU を他処理と競合させると最大 70 ms 前後まで延び、制約の 200 ms を超えるため、受信プロセスには CPU コアを空けておくこと。

### 一致性

- **後段の逐次処理**: 全体を一度に通した場合と一致する（`tools/export_onnx.py` で誤差を検証）。
- **受信全体（`gfm-rx`）**: Python の MendNet + PostNet（一括処理）との差は原音比 **−60 dB 未満**（窓より長い欠落を含む試験では **−45 dB 未満**、`tools/test_rx_onnx.py`）。差の要因は、後段の先読みに補間の暫定値を使う点とストリーム末尾の扱い。

## 送信と復号の対

| 段 | 送信 | 受信 | 入出力 |
|---|---|---|---|
| **変換** | `analyze_stream` | `synthesize_stream` | PCM ⇄ フレームごとの帯域係数 `[M/S][帯域 k][係数 j]`（受信の合成は PostNet グラフ内） |
| **量子化** | `quantize` (`quantize_band`) | `soft_coeffs` | 1 フレームの帯域係数 ⇄ `BlockData`（仮数・指数・アッテネータ）。受信は軟判定で係数と誤差分散を返す |
| **フレーム化** | `pack` | `unpack` | `BlockData` ⇄ 送信ビット列（ヘッダ、FEC、CRC、配置は `Layout`）。純粋関数 |
| **スクランブル** | `scramble` | `descramble` | ビット ⇄ `(LLR, 硬判定)`。同一の PRBS |
| **変調** | `modulate` | `demodulate` | ビット ⇄ シンボル（受信は LLR・硬判定・雑音分散） |
| **フレーム** | `Encoder::frame` | `Decoder::push` | PCM ブロック ⇄ シンボル（受信は欠落を `None` で受ける） |

## 受信の浮動小数点処理

- **LLR とビタビ**: LLR は自然対数の最大対数近似 `(D1 - D0) / (2σ²)`（`σ²` は最寄り信号点との差から推定）。ビタビの経路メトリックも浮動小数点。
- **ヘッダ復号**: 最尤（ビタビ）で基本 CRC を確認し、通らなければ尤度順の候補（リストビタビ、8 候補）から最初に基本 CRC が通るものを採用する。
- **軟判定逆量子化**: 仮数の各ビットが 1 である確率 `p1` は厳密な sigmoid（FEC で訂正した保護ビットは確定値 0/1）。係数は期待値、誤差分散は `(Σ p1(1-p1)·4^pos + 1/12) · 刻み²` の帯域平均。
- **MendNet 入力の LLR**: 自然対数の 4 倍を丸めて `±127` に飽和（学習データと同一）。
- **帯域分割**: 本家と同じリフティング（Haar は S 変換 `d = o − e`、`a = e + d/2`、LeGall 5/3 はブロック跨ぎ）を丸めなしで計算する。
  - 低域の直流利得は 1、どちらも直交ではない。
  - 波形エネルギーは基底エネルギー（`codec::coef_power`。Haar は帯域 `k = 0〜4` で `16, 4, 2, 1, 0.5`）で重み付けする。
  - M/S 逆変換: `L = M + S`、`R = M − S`。

## モジュール一覧

```text
codec/transform.rs   analyze_stream / synthesize_stream、Haar、LeGall 5/3、coef_power
codec/quant.rs       quantize_band / quantize、BlockData
framing.rs           pack / unpack / coefficient_bits
fec.rs               rsc_parity、CRC（送信）、ビタビ・リストビタビ（受信）
scrambler.rs         scramble / descramble
modem.rs             modulate / demodulate
prior.rs             ヘッダの事前知識（Tracker）
encoder.rs           Encoder
decoder.rs           Decoder、soft_coeffs、block_steps
restore.rs           Observation、Mender（MendNet）
gate.rs              Gate（超解像の判定）
post.rs              PostFilter（PostNet）
receiver.rs          Receiver（受信の流れ）
channel.rs           通信路（Rayleigh、欠落、AWGN、12 bit ADC）
dataset.rs           MendNet の学習データ生成（Encoder、通信路、Decoder、Observation）
```
