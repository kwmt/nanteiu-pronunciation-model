# nanteiu-pronunciation-model

英語学習アプリ **なんて言う？（nanteiu）** のオンデバイス発音評価が使う wav2vec2 音素モデルを、
ONNX へ変換・int8 量子化して配布するリポジトリ。

アプリ本体はここには含まれない。ここが持つのは**成果物を作る手順と、その配布先**だけ。

## なぜ本体と分けているか

アプリ本体のリポジトリは非公開だが、**モデルの配布先は公開である必要がある**。
非公開リポジトリの Release アセットは認証なしでは取得できず（認証なしのアクセスは 404 になる）、
アプリに読み取りトークンを埋め込む案は取れない — APK から抽出でき、それは
リポジトリ全体への読み取り権限になってしまう。

配布先が公開であれば、Android も iOS も同じ URL を平文で叩くだけで済む。

## 成果物

[Releases](../../releases) に次の2つを置く。

| ファイル | 用途 |
|---|---|
| `model.onnx` | int8 量子化済み（Conv を除く）の wav2vec2 音素モデル。端末側の ONNX Runtime が推論する |
| `vocab.json` | `{"labels": [...], "blankId": N}`。CTC の出力 id をラベルに戻すための語彙 |

★**`vocab.json` のラベル体系は eSpeak 音素**であり、アプリのサーバー側 g2p（espeak-ng / phonemizer）が
返す正解音素列と**同じ体系でなければならない**。ずれると採点が常に「別音素」判定になって壊れる。
モデルを差し替えるときは、この一致を必ず確認すること。

## 生成手順

CI（[build-model](.github/workflows/build-model.yml)）で回す。手元で回すこともできる。

```bash
pip install -r tools/requirements.txt
python tools/export_wav2vec2_onnx.py --out ./out
```

想定 I/O（端末側の推論実装と一致必須）:

- 入力: `float32[1, samples]`（16kHz の正規化波形）
- 出力: `float32[1, frames, vocab]`（CTC ロジット。softmax は端末側で行う）

## ライセンスと出典

元モデル: [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)（Apache License 2.0）

このリポジトリが配布するのは上記モデルの**派生物**であり、同じく Apache License 2.0 の下で配布する
（[LICENSE](LICENSE)）。加えた変更は次のとおり:

- PyTorch から ONNX 形式へのエクスポート（動的軸: バッチと時間長）
- 重みの int8 動的量子化（端末で動かすためのサイズ・速度の削減）。
  ★**Conv は量子化しない**（`op_types_to_quantize=["MatMul"]`）。既定の設定は Conv を
  `ConvInteger` に置き換えるが、ONNX Runtime Android（ARM64）にその実装が無く、
  端末ではセッションを作る時点で必ず落ちる（nanteiu issue #286）
- 語彙を `{"labels": [...], "blankId": N}` の JSON として書き出し

★**eSpeak-ng（GPLv3）はここには含まれない。** 正解音素列の生成（g2p）はアプリのサーバー側で行い、
端末にも本リポジトリの成果物にも eSpeak-ng を同梱しない（GPL の感染を避けるための設計判断）。
