# engine

GraceFM-Ultra の Rust 実装です。ビット列・フレーム形式は [GraceFM](https://github.com/kmt-t/gracefm) と同一で、音声の符号化と受信は浮動小数点で処理します。

受信は **復号 → MendNet → 判定 → PostNet** をフレームごとに流し、網の推論は ONNX Runtime で動かします（設計詳細は `../docs/DESIGN.md`）。

## クレート構成

| crate | 種別 | 内容 |
|---|---|---|
| `gfm-core` | ライブラリ | 送信（`Encoder`）、受信（`Decoder`, `Mender`, `Gate`, `PostFilter`, `Receiver`）、通信路（`channel`）、学習データ生成（`dataset`） |
| `gfm-tx` | コマンド | PCM を符号化し、通信路（AWGN / Rayleigh / フレーム欠落）を通したシンボルを書き出す |
| `gfm-rx` | コマンド | シンボルを受信して PCM を出力する（ONNX Runtime で MendNet と PostNet を実行） |
| `gfm-gen` | コマンド | MendNet の学習データ（`.npz`）を生成する |

## ビルドと試験

```bash
cargo build --release
cargo test --release
cargo clippy --release --all-targets -- -D warnings
```

- **Python 準拠試験**（本家 Python の受信が `gfm-tx` のシンボルを正しく復号できることの確認）:

```bash
.venv/bin/python tools/test_rs_compat.py   # リポジトリルートで実行
```

## 使い方

受信には **ONNX Runtime（1.31、CPU）の共有ライブラリ** と、**`models/<プロファイル>/`**（`tools/export_onnx.py` が生成する `mendnet.onnx`, `postnet.onnx`）が必要です。ONNX Runtime はリポジトリの `.venv` に含まれるものを利用できます。

```bash
export ORT_DYLIB_PATH=$(ls ../.venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.*)

# 送信 + 通信路
./target/release/gfm-tx --profile B200-L --pcm clip.raw --scen 20,0,0.05,3 --seed 1 --out a.sym

# 受信（--models の既定は models/<プロファイル>）
./target/release/gfm-rx --profile B200-L --frames a.sym --out out.raw

# 学習データ生成
./target/release/gfm-gen --profile B200-L --pcm clip.raw --scen 12,0,0,1 --scen 20,1,0,1 --seed 1 --out data.npz
```

### 引数とファイル形式

| 項目 | 仕様 |
|---|---|
| `clip.raw` | 符号つき 24 bit を `int32`（リトルエンディアン）で格納した L/R 交互の PCM（満点 = 2^23） |
| `--scen e0,fade,rate,burst` | 基準 Es/N0 [dB]、Rayleigh フェージング (0/1)、フレーム欠落率、平均バースト長 [フレーム] |
| `a.sym` | `u32` フレーム数、`u32` フレームあたりのシンボル数、到着フラグ (`u8` × フレーム数)、届いたフレームの I/Q (`f32`、±1 = 1.0) |
| `out.raw` | L/R 交互の `f32`（満点 ±1）、フレーム数 × ブロック長（64）サンプル。時刻は送信と一致（受信機の遅延は処理待ちであり、出力の時刻ずれではない） |

### `gfm-rx` の標準エラー出力

`gfm-rx` は完了時に以下の統計を標準エラー出力へ出力します。

- フレーム統計（`frames`, `lost`, `crc_fail`, `high_crc_fail`, `recovered`）
- 超解像の適用率（`post_on`）と切り替え回数（`switches`）
- 実時間比（`rtf`）と推論処理時間（初期化込みの初回 `first`、残りの `median`, `p99`, `worst`）
- 遅延（`latency` = 補間の確定待ちと合成の 32 kHz で 152 ms + `worst`）
- 最大常駐メモリ（`peak_rss`）

## 対応範囲

- **プロファイル（全 22）**: `B200`, `B200M`, `B200Y`, `B200MY`, `M700`, `M700M`, `M700Y`, `M700MY`, `SUB100`, `SUB50M`, `SUB100-48K` とそれぞれの `-L`（LeGall 5/3）。
  - 本家と同じく Haar と `-L` は `pid` が同一で `format_id` で区別する。`-L` は主系の一部の仮数幅が狭い。
- **受信構成**: 1 系統のみ。モデル（`models/<プロファイル>/`）が必要（同梱は `B200` と `B200-L` のみ、ほかは `tools/train_profile.py` で学習）。
- **未対応**: 暗号化、RTL-SDR の I/Q からの同期（等化器とタイミング同期）。
