# GraceFM-Ultra

[GraceFM](https://github.com/kmt-t/gracefm) のビットストリームを、PC グレードの受信機（RTL-SDR など）で高性能に復号する受信側の実装です。送信側とフレーム形式は変えません。

## 概要

- **構成**: 1 系統受信。フレーム復号のあとに 2 つの DNN（**MendNet** → **PostNet**）を置く。
- **対応プロファイル**: Y 方式・モノラル・SUB・48 kHz を含む本家の全 22 プロファイル。帯域分割も本家と同じく Haar（S 変換）と LeGall 5/3（`-L`）の両方を扱う。
- **モデル**: プロファイルごとに `tools/train_profile.py` で学習する（学習データはリポジトリに含めない）。学習済みモデルは `B200` と `B200-L` のみ `engine/models/` に同梱している。
- **実行環境と性能**（Rust + ONNX Runtime、`engine/`・`docs/DESIGN.md`）:
  - フレームを 1 つずつ逐次処理する。
  - **制約**: 遅延 200 ms 以下、メモリ 2 GB 以下。
  - **実測**（シミュレーションのシンボル入力、32 kHz 系）: 遅延 約 170 ms（確定待ちと合成 152 ms + 計算）、メモリ 約 170 MB。

### 2 つの網の役割

| 網 | 設計書 | 役割 |
|---|---|---|
| **MendNet** | `docs/MENDNET.md` | 前後 100 フレームの文脈・受信 LLR・壊れたフレームの手がかりから、欠落・壊れたフレームを補間し、使えるフレームのビット誤りを補正する。出力は係数（事後平均）と怪しさ（事後標準偏差）。 |
| **PostNet** | `docs/POSTNET.md` | MDCT 領域の超解像。量子化の雑音に埋もれた弱い高域（ピアノの倍音など）を復元する。MendNet の事後標準偏差から出す予測 SNR が高く、元の音質が良いとき（B200 で Es/N0 18 dB、C/N0 約 70.4 dB-Hz あたりから）だけかける。 |

## ターゲットデバイス

受信機は **RTL-SDR Blog V4L (Lite)** で動くように設計する。

| 項目 | 内容 |
|---|---|
| 製品 | RTL-SDR Blog V4L (Lite)、ドングルのみ |
| チューナー | R828S |
| 復調・ADC | RTL2832U（I/Q は 8 bit） |
| 基準発振 | 1 PPM TCXO |
| アンテナ端子 | SMA |
| サンプルレート | 2.4 MS/s 以下（RTL2832U で取りこぼしなく使える範囲） |

### 設計への影響

- **DSP 入力**: 8 bit の I/Q、2.4 MS/s 以下とする。
- **ADC ビット幅の扱い**: 学習データの通信路モデル（`engine/gfm-core/src/channel.rs`）は 12 bit ADC のままとし、実機の 8 bit との差は目標 C/N0 に **1 dB の余裕**を見込んで吸収する。
  - **C/N0 換算**: `C/N0 = Es/N0 + 10 log10(シンボルレート)`（B200 は 172.5 kシンボル/秒で `+52.4 dB`）。評価帯域の Es/N0 16〜30 dB は C/N0 68.4〜82.4 dB-Hz に当たる。
  - **1 dB 余裕の根拠**: 実効 7 bit・AGC の余裕 10 dB とすると、2.4 MS/s 全体の量子化 SNR は約 34 dB。B200 のシンボルレート（172.5 kBd）では Es/N0 約 45 dB に相当し、実用帯の上端（Es/N0 30 dB）でも劣化は 0.1 dB に満たない。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `gracefm_ultra/` | Python コア（本家 import 設定 `gfm.py`、CPU 親和性 `cpus.py`、帯域分割合成 `transform.py`、網 `mendnet.py` / `postnet.py`） |
| `engine/` | Rust 実装（送信・受信・通信路・学習データ生成。詳細は `engine/README.md`）。ONNX は `engine/models/<プロファイル>/` に配置 |
| `tools/` | 学習・評価用スクリプト群 |
| ↳ 音源・データ生成 | `collect_audio.py`（一覧 `audio_sources.json`）、`gen_mend.py`、`gen_post.py` |
| ↳ 学習 | `train_profile.py`（1 プロファイル一括）、`train_mendnet.py`、`train_post.py` |
| ↳ 評価・確認・試聴 | `eval_mendnet.py`、`eval_post.py`、`check_gate.py`、`demo_post.py` |
| ↳ 書き出し・試験 | `export_onnx.py`、`test_rx_onnx.py`、`test_rs_compat.py` |
| `runs/` | 学習・評価のスクリプトとログ（git 管理外） |

## セットアップ

```bash
git clone https://github.com/kmt-t/gracefm.git
git clone https://github.com/kmt-t/gracefm-ultra.git
cd gracefm-ultra
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
(cd engine && cargo build --release)
```

- **受信（Rust）**: Rust + ONNX Runtime で動作する（使い方は `engine/README.md`）。
- **学習・評価（Python）**:
  - 本家 `gracefm` を import する（既定は隣の `../gracefm`、環境変数 `GRACEFM_DIR` で変更可）。
  - 重い処理は環境変数 `ULTRA_CPUS` の CPU コアに固定する（既定は `0,1,5,6,7,11`）。

## 学習から受信まで

### 1. 音源の取得

`tools/audio_sources.json`（Wikimedia Commons の CC0 / パブリックドメイン / CC BY 計 178 本、約 15 GB。作者・ライセンス情報含む）を `data/audio/` に取得する。アクセス間隔を空けて取得するため約 6 時間かかる。User-Agent に入れる連絡先（Wikimedia の規則）を指定すること。

```bash
.venv/bin/python tools/collect_audio.py --contact you@example.com
```

### 2. 学習と ONNX 書き出し

モデルは `engine/models/<プロファイル>/` に書き出される。

```bash
# 1 プロファイルを通しで実行（data → mendnet → postdata → postnet → export → verify）
.venv/bin/python tools/train_profile.py B200-L

# 全プロファイルを学習し、完了後に中間データを削除
.venv/bin/python tools/train_profile.py --all --clean

# 短いデータと短い学習で動作確認
.venv/bin/python tools/train_profile.py M700Y --quick --data /tmp/q --runs /tmp/q/runs --models /tmp/q/models
```

- **計算量・ディスク**: 1 プロファイルあたり GPU で約 9 時間、ディスクは最大 `127 GB × (標本化周波数 / 32 kHz)` を使用する。
- **オプション**: `--stages` で実行段を選べる（`eval` は明示指定時のみ実行）。`postdata` 段では判定閾値の確認（`tools/check_gate.py`）も併せて出力する。

## 設計ドキュメント

- [docs/DESIGN.md](docs/DESIGN.md): 受信機（Rust）の構成、遅延、送信と復号の対
- [docs/MENDNET.md](docs/MENDNET.md): MendNet（補間・ビット誤り補正）、ヘッダの事前知識つき復号とリスト復号
- [docs/POSTNET.md](docs/POSTNET.md): PostNet（MDCT 領域の超解像）、判定、評価

## ライセンス

[PolyForm Noncommercial License 1.0.0](LICENSE)（GraceFM と同じ）。`NOTICE` の Required Notice を保持すること。GraceFM に依存するため、その条件にも従うこと。
