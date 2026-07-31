#!/usr/bin/env python3
"""
wav2vec2 音素モデル（facebook/wav2vec2-lv-60-espeak-cv-ft）を ONNX に変換し int8 量子化する。

このスクリプトは【この開発環境では実行しない】。Python + PyTorch + transformers + onnxruntime が
要る重い処理で、成果物（約200MB の .onnx と vocab.json）を生成する。実行は user / CI で行う。

生成物:
  - model.onnx        … 量子化済みモデル（core:asr の OnnxPhonemeRecognizer が想定する I/O）
  - vocab.json        … {"labels": [...], "blankId": N}（core:asr の OnnxModelProvider が読む形式）

想定 I/O（OnnxPhonemeRecognizer.kt と一致させること）:
  - 入力: float32[1, samples]  （16kHz 正規化波形。入力名は動的軸 'samples'）
  - 出力: float32[1, frames, vocab]  （CTC ロジット。Kotlin 側で softmax する）

使い方:
  pip install torch transformers onnx onnxruntime optimum
  python export_wav2vec2_onnx.py --out ./out

生成後:
  - out/model.onnx を CDN / GitHub Releases / Render 静的配信 等にホスティングし、
    URL を core:asr の OnnxModelSource（DI）に渡す
  - out/vocab.json も同様にホスティング
"""
import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./out", help="出力ディレクトリ")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--no-quantize", action="store_true", help="int8 量子化をスキップ")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = AutoModelForCTC.from_pretrained(args.model)
    model.eval()

    # 1) vocab.json を core:asr の期待形式で書き出す（labels は id 順、blankId は pad token）。
    #    ★AutoProcessor/Wav2Vec2PhonemeCTCTokenizer は phonemizer + espeak-ng を要求するが、
    #      export に必要なのは vocab（ラベル↔id）と pad token だけ。HF リポジトリの vocab.json /
    #      tokenizer_config.json を直接読み、phonemizer 依存を避ける（CI を軽く・堅くする）。
    with open(hf_hub_download(args.model, "vocab.json"), encoding="utf-8") as f:
        vocab_map = json.load(f)  # {label: id}
    id_to_label = {i: t for t, i in vocab_map.items()}
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    # blank（CTC）は pad token。tokenizer_config.json の pad_token（既定 "<pad>"）の id を使う
    pad_token = "<pad>"
    try:
        with open(hf_hub_download(args.model, "tokenizer_config.json"), encoding="utf-8") as f:
            pad_token = json.load(f).get("pad_token", "<pad>")
    except Exception:
        pass
    blank_id = vocab_map.get(pad_token, vocab_map.get("<pad>", 0))
    with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "blankId": blank_id}, f, ensure_ascii=False)

    # 2) ONNX 変換（入力 [1, samples] 動的軸、出力 [1, frames, vocab]）
    dummy = torch.randn(1, 16000)  # 1秒ぶんのダミー波形
    onnx_fp32 = os.path.join(args.out, "model_fp32.onnx")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_fp32,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "samples"},
            "logits": {0: "batch", 1: "frames"},
        },
        opset_version=17,
    )

    final = os.path.join(args.out, "model.onnx")
    if args.no_quantize:
        os.replace(onnx_fp32, final)
    else:
        # 3) int8 動的量子化（約377MB → 約200MB）
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(onnx_fp32, final, weight_type=QuantType.QInt8)
        os.remove(onnx_fp32)

    size_mb = os.path.getsize(final) / (1024 * 1024)
    print(f"done: {final} ({size_mb:.1f} MB), vocab labels={len(labels)}, blankId={blank_id}")
    print("この model.onnx / vocab.json をホスティングし、URL を OnnxModelSource に渡すこと。")


if __name__ == "__main__":
    main()
