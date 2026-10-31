# GraceFM-Ultra

[GraceFM](../gracefm) のビットストリームを、PCグレードの受信機（RTL-SDRなど）で高性能に復号する受信側の実装。送信側とフレーム形式は変えない。

劣化モードはビット反転とフレーム欠損。それぞれ別のモデルで扱う。

- **HaarNet-lite**（ビット反転）: 全帯域の仮数bitのLLRと刻みから、Haar係数を非因果に推定する。生LLRの事後平均 E0 に対する補正量を、飛び幅つき畳み込み（TCN）が出す。プロファイルごとに学習する。
- **PlcNet**（欠落補間）: 前後の文脈から欠落フレームを補間する。両端クロスフェードと、欠落区間だけの因果IIR（Butterworth 4次 4 kHz）切替で、Haar基底の漏れによる高域の広帯域雑音を抑える。

制約: バッファ200 msまで、メモリ2 GBまで。エンジンの最終形は Rust + ONNX Runtime（未着手）。

## baseline

`runs/baseline/`（`runs/baseline/README.md` に作り方と評価）。改善は、ここを基準にして比べる。

## ディレクトリ

- `gracefm_ultra/`: 受信（`rx.py`）、HaarNet（`haarnet.py`）、PlcNet（`plcnet.py`）、IIR切替（`iir_switch.py`）
- `tools/`: データ生成（`gen_dataset.py`、`prep_haarnet.py`）、学習（`train_haarnet.py`、`train_plc.py`）、評価（`eval_*.py`、`baseline.py`）、デモ・測定
- `runs/`: 学習・評価のスクリプトとログ（モデルと一部ログは git 管理外）

## ライセンス

[PolyForm Noncommercial License 1.0.0](LICENSE)（GraceFMと同じ）。`NOTICE` の Required Notice を保持すること。GraceFMに依存するため、その条件にも従う。
