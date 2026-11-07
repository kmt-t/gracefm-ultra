# 受信機の設計（Rust、`engine/gfm-core`）

GraceFM のビット列・フレーム形式は変えない。送信は本家 Python（`../gracefm`）と鏡になる構成で、受信は復号のあとに MendNet と PostNet を置く。

- **帯域分割はプロファイルで選ぶ**: 本家と同じく Haar（S 変換、format_id 3）と LeGall 5/3（`-L`、format_id 2）。網はプロファイルごとに学習する。本家の受信の出力段（PLC、IIR 切替、再標本化）は使わず、PostNet の出力をプロファイルの標本化周波数のまま出す。
- **音声の符号化と受信は浮動小数点**: DSP では動かさず、音質を優先する。本家の整数の演算幅の仕様は持ち込まないので、本家とはビット単位では一致しない。
- **ビット配置・FEC・CRC・スクランブルはビット処理**: 本家とビット単位で同じ。Rust の送信を本家 Python の受信で復号する試験（`tools/test_rs_compat.py`）で確かめる。
- **網は ONNX**: 入力の特徴づくりと出力の後処理もグラフに入れる（`tools/export_onnx.py`）。Rust は観測と状態を並べて渡すだけ。

## 受信の流れ

フレームを 1 つずつ入れる。各段は、次の段に渡すものだけを出す。

```
シンボル ─▶ Decoder ─DecodedFrame─▶ Observation ─▶ Mender ─Mended─▶ Gate ─(Restored, 判定)─▶ PostFilter ─▶ 音声 L/R
            復号                     網の入力の符号化   MendNet          判定             PostNet
```

| 段 | 型 | 受け持つこと | 出すもの |
|---|---|---|---|
| 復号 | `decoder::Decoder` | 復調、スクランブル解除、ヘッダ（事前知識つき、通らなければリスト復号）、FEC、軟判定の逆量子化。直前までのヘッダを覚える | `DecodedFrame`: 状態、帯域係数、帯域ごとの誤差分散、仮数ビットの受信 LLR、ヘッダ、gap、壊れたフレームの手がかり |
| 観測 | `restore::Observation` | MendNet の入力の定義どおりに符号化する（LLR の量子化、刻み、情報なしは −1） | 1 フレームの観測 |
| 補間 | `restore::Mender` | 直近 100 フレームの窓を 50 フレームごとに MendNet にかけ、中央 50 フレームを確定する（最初の窓は先頭から） | `Mended`: 確定したフレーム（e0、sd、元が使えたか）と、窓の残りの暫定のフレーム |
| 判定 | `gate::Gate` | 過去 1 秒の、使えたフレームの e0 の波形でのエネルギー / sd の同じもの（予測の SNR）。27.5 dB 以上でオン、オンの間は 25.5 dB を下回るまでオン | フレームごとの判定 |
| 後段 | `post::PostFilter` | PostNet を段ごとの状態を持ち越して逐次に回す。確定したフレームだけで決まる位置まで状態を進め、暫定のフレームに頼る先読みの部分は毎回計算し直す | 音声（L/R、満点 ±1） |
| 受信 | `receiver::Receiver` | 上をフレームごとにつなぐ。超解像をかけた割合と切り替えの回数を数える | 音声 |

網の推論は `restore::MendModel` と `post::PostModel` に任せる。`gfm-rx` が ONNX Runtime で実装し、試験（`tests/stream.rs`）は入力をそのまま返す模型で窓と区間の受け渡しを確かめる。

### 遅延

- 補間の確定待ち: 窓 [a, a + 100) はフレーム a + 99 が届いた時点で、[a + 25, a + 75) を確定する。最も待つフレームは 75 フレーム（150 ms）。
- 合成の届く範囲: フレームの末尾の標本には次のフレームの係数が届くので、音声は確定したフレームより 1 フレーム手前まで出す（2 ms）。暫定の係数が出力に混ざって、確定の境界に継ぎ目ができるのを防ぐ。
- 後段の先読み（MDCT 2 フレーム + 窓の重なり + 合成に要る隣の 1 フレーム、最大 16 フレーム）は、補間の窓の残り（25 フレーム）の暫定値で賄うので、待ちは増えない。
- `-L` は、送信の符号化に 15 サンプル（0.47 ms）の先読みが要る。Haar はどちらも 0。
- 計算: 50 フレームごとに MendNet 1 回（窓 1 つ × M・S）と PostNet 1 歩。
- 合計は、152 ms + 1 回の計算の時間。`gfm-rx` が実測を出す（`latency`）。
- 実測（B200 と B200-L、30 秒、Es/N0 24 dB、フレーム欠落 5%、推論 2 スレッド、ほかの処理のない CPU 2 個、各 4 回）: 計算は中央値 10 ms、最大 12〜20 ms で、遅延は 164〜172 ms。実時間比 0.14、メモリ 169 MB。ファイルから読んで書く処理で、実時間の入出力は含まない。
- CPU を他の処理と取り合うと、最大 70 ms 前後まで延びて 200 ms を超える。受信には CPU を空けておく。
- シミュレーションのシンボルファイルからの測定で、RTL-SDR の I/Q の処理（同期、等化）と音声出力のバッファは含まない。

### 一致

- 後段の逐次処理は、全体を一度に通したものと一致する（`tools/export_onnx.py` が差を出す）。
- 受信全体（`gfm-rx`）は、Python の MendNet + PostNet（全体を一度に処理）と、原音比 −60 dB 未満の差で一致する（`tools/test_rx_onnx.py`）。差は、後段の先読みに補間の暫定値を使うぶんと、ストリームの末尾の扱い。

## 送信と復号の対

| 段 | 送信 | 受信 | 入出力 |
|---|---|---|---|
| 変換 | `analyze_stream` | `synthesize_stream` | PCM ⇄ フレームごとの帯域係数 `[成分 M/S][帯域 k][係数 j]`（M/S 分離と帯域分割をまとめる。受信の合成は PostNet のグラフの中） |
| 量子化 | `quantize`（`quantize_band`） | `soft_coeffs` | 1 フレームの帯域係数 ⇄ `BlockData`（仮数・指数・アッテネータ）。受信は軟判定で、係数と帯域ごとの誤差分散を返す |
| フレーム化 | `pack` | `unpack` | `BlockData` ⇄ 送信ビット列（ヘッダ、FEC、CRC、配置は `Layout`）。どちらも純粋な関数 |
| スクランブル | `scramble` | `descramble` | ビット ⇄ （LLR, 硬判定）。同じ PRBS |
| 変調 | `modulate` | `demodulate` | ビット ⇄ シンボル（受信は LLR・硬判定・雑音分散） |
| フレーム | `Encoder::frame` | `Decoder::push` | PCM ブロック ⇄ シンボル（受信は欠落を `None` で受ける） |

## 受信の浮動小数点の中身

- LLR は自然対数の最大対数近似 `(D1 - D0) / (2σ²)`、σ² は最寄りの信号点との差から。ビタビの経路メトリックも浮動小数点。
- ヘッダは、まず最尤（ビタビ）で基本 CRC を確かめ、通らなければ尤度の高い順の候補（リストビタビ、8 候補）から最初に通るものを採る。
- 仮数の各ビットが 1 である確率 p1 は厳密な sigmoid。FEC で訂正した保護ビットは確定値（0 か 1）。係数は期待値、誤差分散は `Σ p1(1-p1)·4^pos + 1/12`（刻みの二乗倍）の帯域平均。
- MendNet の入力の受信 LLR は、自然対数の 4 倍を丸めて ±127 に飽和（学習データと同じ）。
- 帯域分割は本家と同じリフティング（Haar は S 変換 `d = o − e`、`a = e + d/2`。LeGall 5/3 はブロックをまたぐ）を丸めなしで計算する。低域の直流の利得は 1 で、どちらも直交ではない。波形でのエネルギーは、基底のエネルギー（`codec::coef_power`。Haar は帯域 k = 0〜4 で 16、4、2、1、0.5）で重みを付ける。L = M + S、R = M − S。

## モジュール

```
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
