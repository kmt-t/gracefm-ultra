# GraceFM-Ultra

[GraceFM](../gracefm) のビットストリームを、PCグレードの受信機（RTL-SDRなど）で高性能に復号する受信側の設計・実装。送信側とフレーム形式は変えない。

劣化モードがビット反転とフレーム欠損に限られることを利用し、次の3つを一つのベイズ推定にまとめる。

- **PriorNet**: LPC外挿の深層学習版。前後最大200 msの文脈から各Haar係数の分布を予測する。欠落補間と事前分布に使う。
- **BitNet**: ビット反転判定の深層学習版。軟値と文脈から量子化indexの事後分布を出す。
- **Wiener超解像**: 量子化・誤りによる雑音と事前分散からWiener利得を決め、最終推定を行う。

制約: バッファ200 msまで、メモリ2 GBまで。現状は設計段階で、実装・測定結果はまだない。

## ライセンス

[PolyForm Noncommercial License 1.0.0](LICENSE)（GraceFMと同じ）。`NOTICE` の Required Notice を保持すること。GraceFMに依存するため、その条件にも従う。
