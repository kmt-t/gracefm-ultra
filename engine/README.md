# engine

GraceFM-Ultra の Rust 実装。ビット列・フレーム形式は [GraceFM](../../gracefm) と同じで、音声の符号化と受信は浮動小数点で書く。
受信は、復号 → MendNet → 判定 → PostNet をフレームごとに流す（`../docs/DESIGN.md`）。網は ONNX Runtime で動かす。

| crate | 内容 |
|---|---|
| `gfm-core` | ライブラリ。送信（`Encoder`）、受信（`Decoder`、`Mender`、`Gate`、`PostFilter`、`Receiver`）、通信路（`channel`）、学習データ生成（`dataset`） |
| `gfm-tx` | コマンド。PCM を符号化し、通信路（AWGN / Rayleigh / フレーム欠落）を通したシンボルを書く |
| `gfm-rx` | コマンド。シンボルを受信して PCM にする（ONNX Runtime で MendNet と PostNet を動かす） |
| `gfm-gen` | コマンド。MendNet の学習データ（.npz）を作る |

## ビルドと試験

```
cargo build --release
cargo test --release
cargo clippy --release --all-targets -- -D warnings
```

Python 側の準拠試験（本家の Python が、`gfm-tx` のシンボルを正しく復号できること）: `.venv/bin/python tools/test_rs_compat.py`

## 使い方

受信には ONNX Runtime（1.31、CPU）の共有ライブラリと、`models/B200/`（`tools/export_onnx.py` が書く mendnet.onnx、postnet.onnx）を使う。
ONNX Runtime は、リポジトリの `.venv` の onnxruntime（pip）に入っているものを使える。

```
export ORT_DYLIB_PATH=$(ls ../.venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.*)
gfm-tx --profile B200 --pcm clip.raw --scen 20,0,0.05,3 --seed 1 --out a.sym      # 送信 + 通信路
taskset -c 0,1,5,6,7,11 gfm-rx --profile B200 --frames a.sym --out out.raw --models models/B200   # 受信
gfm-gen --profile B200 --pcm clip.raw --scen 12,0,0,1 --scen 20,1,0,1 --seed 1 --out data.npz   # 学習データ
```

- `clip.raw`: 符号つき 24 bit を int32（リトルエンディアン）で入れた L/R の交互（満点 = 2^23）
- `--scen e0,fade,rate,burst`: 基準 Es/N0 [dB]、Rayleigh か (0/1)、フレーム欠落率、平均バースト長 [フレーム]
- `a.sym`: u32 フレーム数, u32 フレームあたりのシンボル数, フレーム数個の到着フラグ (u8), 届いたフレームの I/Q (f32、±1 = 1.0)
- `out.raw`: L/R の交互の f32（満点 ±1）、フレーム数 × 64 サンプル。時刻は送信と揃っている（受信機の遅延は、出力の時刻ではなく、処理の待ち）
- `gfm-rx` は標準エラー出力に、フレームの統計、超解像をかけた割合（`post_on`）と切り替えの回数、実時間比（`rtf`）、1 フレームの処理の最大時間（`worst_push`）、遅延（`latency` = 補間の確定待ち 150 ms + `worst_push`）、最大の常駐メモリを書く

## 対応範囲

- 送信: B200、B200M、M700、M700M、SUB100、SUB50M、SUB100-48K
- 受信: B200（MendNet と PostNet を学習したプロファイル）、1 系統
- 未対応: Y 方式（`*Y`）、暗号化、RTL-SDR の I/Q からの同期（等化器とタイミング同期）
