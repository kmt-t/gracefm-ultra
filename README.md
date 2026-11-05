# GraceFM-Ultra

[GraceFM](../gracefm) のビットストリームを、PCグレードの受信機（RTL-SDRなど）で高性能に復号する受信側の実装。送信側とフレーム形式は変えない。

劣化モードはビット反転とフレーム欠損。それぞれ別のモデルで扱う。

- **HaarNet-lite**（ビット反転）: 全帯域の仮数bitのLLRと刻みから、Haar係数を非因果に推定する。生LLRの事後平均 E0 に対する補正量を、飛び幅つき畳み込み（TCN）が出す。プロファイルごとに学習する。
- **MendNet**（欠落補間）: 前後の文脈から欠落フレームを補間する。両端クロスフェードと、欠落区間だけの因果IIR（Butterworth 4次 4 kHz）切替で、Haar基底の漏れによる高域の広帯域雑音を抑える。

制約: バッファ200 msまで、メモリ2 GBまで。エンジンの最終形は Rust + ONNX Runtime（未着手）。

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

## baseline

`runs/baseline/`（`runs/baseline/README.md` に作り方と評価）。改善は、ここを基準にして比べる。

## ディレクトリ

- `gracefm_ultra/`: 本家 GraceFM を import する設定（`gfm.py`）、HaarNet（`haarnet.py`）、MendNet（`mendnet.py`）、IIR切替（`iir_switch.py`）
- `engine/`: Rust 実装（送信、受信、通信路、学習データ生成。`engine/README.md`）
- `tools/`: データ生成（`gen_hn.py`、Rust の `engine/gfm-gen`）、学習（`train_haarnet.py`、`train_mendbits.py`）、評価（`eval_*.py`、`baseline.py`）、デモ・測定
- `runs/`: 学習・評価のスクリプトとログ（モデルと一部ログは git 管理外）

## 設計

- `docs/DESIGN.md`: Rust エンジン（送信と受信の対称な構成）
- `docs/MENDNET.md`: MendNet（欠落・壊れたフレームの補間）、副系を使うヘッダの復元

## ライセンス

[PolyForm Noncommercial License 1.0.0](LICENSE)（GraceFMと同じ）。`NOTICE` の Required Notice を保持すること。GraceFMに依存するため、その条件にも従う。
