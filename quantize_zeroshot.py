import argparse
import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import BertOnnxConfig
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
OUTPUT_PATH = "./onnx_model"

def export_to_onnx(model_id, out_dir):
    print("🔄 Exporting to ONNX…")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Use BERT-style config for MiniLM
    onnx_config = BertOnnxConfig(model.config)

    output_path = Path(out_dir) / "model.onnx"  # ✅ Create a Path object
    export(
        model=model,
        config=onnx_config,
        output=output_path,   # ✅ Use the Path object here
        opset=13,
    )
    return tokenizer

def quantize(fp32_path, int8_path):
    print("🔧 Quantizing to INT8…")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8
    )
    print(f"✅ Saved quantized model to {int8_path}")

def quick_smoke_test(int8_path, model_id, text, label):
    print("🧪 Smoke test…")
    tok = AutoTokenizer.from_pretrained(model_id)
    hyp = f"This text is about {label}."
    enc = tok(text, hyp, return_tensors="np", truncation=True, padding=True, max_length=512)

    onnx_inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    }
    if "token_type_ids" in enc:
        onnx_inputs["token_type_ids"] = enc["token_type_ids"]

    sess = ort.InferenceSession(int8_path)
    logits = sess.run(None, onnx_inputs)[0]

    print("logits shape:", logits.shape)
    print("logits:", logits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", default=OUTPUT_PATH)
    args = parser.parse_args()

    out_fp32 = os.path.join(args.out, "model.onnx")
    out_int8 = os.path.join(args.out, "model-quant.onnx")

    tok = export_to_onnx(args.model, args.out)
    quantize(out_fp32, out_int8)
    quick_smoke_test(out_int8, args.model, "This shipping sucks", "delivery issue")

if __name__ == "__main__":
    main()
