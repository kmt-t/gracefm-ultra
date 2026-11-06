# GraceFM-Ultra

[GraceFM](../gracefm) のビットストリームを、PCグレードの受信機（RTL-SDRなど）で高性能に復号する受信側の実装。送信側とフレーム形式は変えない。

受信は、フレームの復号のあとに 2 つの網を置く（B200 プロファイル、1 系統）。

- **MendNet**（`docs/MENDNET.md`）: 前後 100 フレームの文脈、受信 LLR、壊れたフレームに残る手がかりから、欠落・壊れたフレームを補間し、使えるフレームのビット誤りを補正する。係数（事後平均）と、その怪しさ（事後標準偏差）を出す。
- **PostNet**（`docs/POSTNET.md`）: MDCT 領域の超解像。Haar の量子化の雑音の床に沈んだ弱い高域（ピアノの倍音など）を復元する。MendNet の事後標準偏差から出す予測の SNR で、元の音質が良いとき（C/N0 68 dB-Hz あたりから）だけかける。

受信機は Rust + ONNX Runtime（`engine/`、`docs/DESIGN.md`）。フレームを 1 つずつ受け、遅延は 150 ms + 計算（1 回あたり数十 ms）。制約: 遅延 200 ms まで、メモリ 2 GB まで。

## ターゲットデバイス

受信機は **RTL-SDR Blog V4L (Lite)** で動くように設計する。

| 項目 | 内容 |
|---|---|
| 製品 | RTL-SDR Blog V4L (Lite)、ドングルのみ |
| チューナー | R828S |
| 復調・ADC | RTL2832U。I/Q は 8 bit |
| 基準発振 | 1 PPM TCXO |
| アンテナ端子 | SMA |
| サンプルレート | 2.4 MS/s 以下（RTL2832U で取りこぼしなく使える範囲） |

設計への影響:
- 受信の DSP の入力は、8 bit の I/Q で 2.4 MS/s 以下とする。
- 学習データの通信路のモデル（`engine/gfm-core/src/channel.rs`）の ADC は 12 bit のままとし、8 bit との差は C/N0 に 1 dB の下駄を履かせて見込む。実機の C/N0 66〜80 dB-Hz（実用帯）は、モデルの Es/N0 15〜29 dB（B200）に当たる。
- 下駄の根拠: 実効 7 bit、AGC の余裕 10 dB として、2.4 MS/s 全体の量子化の SNR は約 34 dB。B200 のシンボル帯域（100 kBd）では Es/N0 約 48 dB に当たり、実用帯の上端（Es/N0 30 dB）でも劣化は 0.1 dB に満たない。1 dB はその上の余裕。

## ディレクトリ

- `gracefm_ultra/`: 本家 GraceFM を import する設定（`gfm.py`）、MendNet（`mendnet.py`）、PostNet（`postnet.py`）
- `engine/`: Rust 実装（送信、受信、通信路、学習データ生成。`engine/README.md`）。`engine/models/B200/` に受信が使う ONNX
- `tools/`: 音源の収集（`collect_audio.py`）、データ生成（`gen_mend.py`、`gen_post.py`）、学習（`train_mendbits.py`、`train_post.py`）、評価（`eval_mendnet.py`、`eval_post.py`）、ONNX の書き出し（`export_onnx.py`）、試験（`test_rx_onnx.py`、`test_rs_compat.py`）、試聴（`demo_post.py`）
- `runs/`: 学習・評価のスクリプトとログ（git 管理外）

## 学習から受信まで

```
.venv/bin/python tools/gen_mend.py --profile B200 --split train --sub mend          # MendNet の学習データ（gfm-gen）
.venv/bin/python tools/train_mendbits.py --profile B200 --out runs/mendbits_B200.pt    # MendNet
.venv/bin/python tools/gen_post.py --ckpt runs/mendbits_B200_v9.pt --src mend --split train --out train   # PostNet の学習データ
.venv/bin/python tools/train_post.py --profile B200 --out runs/post_B200.pt           # PostNet
.venv/bin/python tools/export_onnx.py --mend runs/mendbits_B200_v9.pt --post runs/post_B200.pt --out engine/models/B200
.venv/bin/python tools/test_rx_onnx.py --mend runs/mendbits_B200_v9.pt --post runs/post_B200.pt   # Rust の受信と Python の一致
```

## 設計

- `docs/DESIGN.md`: 受信機（Rust）の構成、遅延、送信と復号の対
- `docs/MENDNET.md`: MendNet（欠落・壊れたフレームの補間、ビット誤りの補正）、ヘッダの事前知識つき復号とリスト復号
- `docs/POSTNET.md`: PostNet（MDCT 領域の超解像）、判定、評価

## ライセンス

[PolyForm Noncommercial License 1.0.0](LICENSE)（GraceFMと同じ）。`NOTICE` の Required Notice を保持すること。GraceFMに依存するため、その条件にも従う。
