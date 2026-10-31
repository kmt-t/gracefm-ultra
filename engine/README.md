# engine

GraceFM-Ultra の Rust 実装。ビット列・フレーム形式は [GraceFM](../../gracefm) と同じで、音声の符号化と受信は浮動小数点（f64）で書く。
Python 版 GraceFM（本家）と構成を合わせてある（`docs/DESIGN.md`）。

| crate | 内容 |
|---|---|
| `gfm-core` | ライブラリ。送信（`Encoder`）、受信（`Decoder`、`receiver`）、通信路（`channel`）、学習データ生成（`gen`） |
| `gfm-tx` | コマンド。PCM を符号化し、通信路（AWGN / Rayleigh / フレーム欠落）を通したシンボルを書く |
| `gfm-rx` | コマンド。シンボルを受信して PCM にする（Ultra の受信） |
| `gfm-gen` | コマンド。HaarNet の学習データ（.npz）を作る |

## ビルドと試験

```
cargo build --release
cargo test --release
cargo clippy --release --all-targets -- -D warnings
```

Python 側の準拠試験（本家の Python が、`gfm-tx` のシンボルを正しく復号できること）: `.venv/bin/python tools/test_rs_compat.py`

## 使い方

```
gfm-tx --profile B200 --pcm clip.raw --scen 12,0,0.1,4 --seed 1 --out a.sym     # 送信 + 通信路
gfm-rx --profile B200 --frames-a a.sym --n 64000 --out out.raw                    # 受信
gfm-gen --profile B200 --pcm clip.raw --scen 12,0,0,1 --scen 20,1,0,1 --seed 1 --out data.npz   # 学習データ
```

- `clip.raw`: 符号つき 24 bit を int32（リトルエンディアン）で入れた L/R の交互（満点 = 2^23）
- `--scen e0,fade,rate,burst`: 基準 Es/N0 [dB]、Rayleigh か (0/1)、フレーム欠落率、平均バースト長 [フレーム]
- `a.sym`: u32 フレーム数, u32 フレームあたりのシンボル数, フレーム数個の到着フラグ (u8), 届いたフレームの I/Q (f32、±1 = 1.0)
- `out.raw`: L/R の交互の f32（満点 ±1）。出力段の IIR のぶん、3 サンプル（32 kHz 系）か 5 サンプル（48 kHz 系）遅れる

## 対応範囲

- プロファイル: B200、B200M、M700、M700M、SUB100、SUB50M、SUB100-48K
- 暫定: 欠落補間は LPC 外挿（`plc.rs`）。本番は PlcNet（DNN、聴感のため）で、ONNX Runtime で載せる予定。載せたら LPC は削除する
- 未対応: Y 方式（`*Y`）、暗号化、DAC 前のポストフィルタ、等化器とタイミング同期
