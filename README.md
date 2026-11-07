# GraceFM-Ultra

[GraceFM](https://github.com/kmt-t/gracefm) のビットストリームを、PCグレードの受信機（RTL-SDRなど）で高性能に復号する受信側の実装。送信側とフレーム形式は変えない。

受信は、フレームの復号のあとに 2 つの網を置く（1 系統）。本家と同じく Haar（S 変換）と LeGall 5/3（名前の末尾が `-L`）の両方の帯域分割を扱い、網はプロファイルごとに学習する。受信は本家の全 22 プロファイル（Y 方式、モノラル、SUB、48 kHz を含む）を扱い、学習は `tools/train_profile.py` で通す。学習済みのモデルは B200 と B200-L だけ `engine/models/` に入れている。ほかのプロファイルは、音源を取って学習する（「学習から受信まで」）。学習データはリポジトリに入れない。

- **MendNet**（`docs/MENDNET.md`）: 前後 100 フレームの文脈、受信 LLR、壊れたフレームに残る手がかりから、欠落・壊れたフレームを補間し、使えるフレームのビット誤りを補正する。係数（事後平均）と、その怪しさ（事後標準偏差）を出す。
- **PostNet**（`docs/POSTNET.md`）: MDCT 領域の超解像。量子化の雑音の床に沈んだ弱い高域（ピアノの倍音など）を復元する。MendNet の事後標準偏差から出す予測の SNR で、元の音質が良いとき（B200 で Es/N0 18 dB、C/N0 70 dB-Hz あたりから）だけかける。

受信機は Rust + ONNX Runtime（`engine/`、`docs/DESIGN.md`）。フレームを 1 つずつ受け、制約は遅延 200 ms まで、メモリ 2 GB まで。シミュレーションのシンボルからの処理で、遅延は約 170 ms（補間の確定待ちと合成 152 ms + 計算）、メモリは約 170 MB（`docs/DESIGN.md`）。

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
- 学習データの通信路のモデル（`engine/gfm-core/src/channel.rs`）の ADC は 12 bit のままとし、8 bit との差は C/N0 に 1 dB の下駄を履かせて見込む。C/N0 は本家と同じく Es/N0 にシンボルレートの dB を足した値で、B200 は 172.6 kシンボル/秒で +52.4 dB。評価する Es/N0 16〜30 dB は C/N0 68.4〜82.4 dB-Hz に当たる。
- 下駄の根拠: 実効 7 bit、AGC の余裕 10 dB として、2.4 MS/s 全体の量子化の SNR は約 34 dB。B200 のシンボルレート（172.6 kBd）では Es/N0 約 45 dB に当たり、実用帯の上端（Es/N0 30 dB）でも劣化は 0.1 dB に満たない。1 dB はその上の余裕。

## ディレクトリ

- `gracefm_ultra/`: 本家 GraceFM を import する設定（`gfm.py`）、帯域分割の合成（`transform.py`）、MendNet（`mendnet.py`）、PostNet（`postnet.py`）
- `engine/`: Rust 実装（送信、受信、通信路、学習データ生成。`engine/README.md`）。受信が使う ONNX は `tools/export_onnx.py` が `engine/models/<プロファイル>/` に書く
- `tools/`: 音源の取得（`collect_audio.py`、一覧は `audio_sources.json`）、データ生成（`gen_mend.py`、`gen_post.py`）、学習（`train_mendnet.py`、`train_post.py`）、評価（`eval_mendnet.py`、`eval_post.py`）、ONNX の書き出し（`export_onnx.py`）、試験（`test_rx_onnx.py`、`test_rs_compat.py`）、試聴（`demo_post.py`）、プロファイルごとの学習の一括実行（`train_profile.py`）、判定の閾値の確認（`check_gate.py`）
- `runs/`: 学習・評価のスクリプトとログ（git 管理外）

## 準備

```
git clone https://github.com/kmt-t/gracefm.git
git clone https://github.com/kmt-t/gracefm-ultra.git
cd gracefm-ultra
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
(cd engine && cargo build --release)
```

受信は Rust と ONNX Runtime で動く（`engine/README.md` の使い方）。学習と評価の Python は本家の実装を import する。場所は隣の `gracefm` で、環境変数 `GRACEFM_DIR` で変えられる。Python の重い処理は環境変数 `ULTRA_CPUS` の CPU に固定する（既定は `0,1,5,6,7,11`。別のマシンでは指定する）。

## 学習から受信まで

まず音源を `data/audio/` に取る。一覧は `tools/audio_sources.json`（Wikimedia Commons の CC0、パブリックドメイン、CC BY の 178 本、約 15 GB。作者とライセンスもここにある）で、間隔を空けて取るので 6 時間ほどかかる。連絡先は Wikimedia の規則で User-Agent に入れる。

```
.venv/bin/python tools/collect_audio.py --contact you@example.com
```

次に学習する。モデルは `engine/models/<プロファイル>/` に書かれる。

```
.venv/bin/python tools/train_profile.py B200-L              # 1 プロファイルを通す: data → mendnet → postdata → postnet → export → verify
.venv/bin/python tools/train_profile.py --all --clean       # 全プロファイル。終わったらデータを消す
.venv/bin/python tools/train_profile.py M700Y --quick --data /tmp/q --runs /tmp/q/runs --models /tmp/q/models   # 動作確認（短いデータと短い学習）
```

1 プロファイルに GPU で約 9 時間と、最大 127 GB × (標本化周波数 / 32 kHz) のディスクを使う。`--stages` で段を選べる（`eval` は指定したときだけ）。`postdata` の段は判定の閾値の確認（`tools/check_gate.py`）も出す。

## 設計

- `docs/DESIGN.md`: 受信機（Rust）の構成、遅延、送信と復号の対
- `docs/MENDNET.md`: MendNet（欠落・壊れたフレームの補間、ビット誤りの補正）、ヘッダの事前知識つき復号とリスト復号
- `docs/POSTNET.md`: PostNet（MDCT 領域の超解像）、判定、評価

## ライセンス

[PolyForm Noncommercial License 1.0.0](LICENSE)（GraceFMと同じ）。`NOTICE` の Required Notice を保持すること。GraceFMに依存するため、その条件にも従う。
