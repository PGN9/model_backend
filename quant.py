import argparse
import os
from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np


def export_to_onnx(model_id: str, output_path: str):
    print(f"🔄 Exporting model '{model_id}' to ONNX...")
    main_export(
        model_name_or_path=model_id,
        output=output_path,
        task="feature-extraction",  # Changed from "text-classification"
        framework="pt"
    )
    print("✅ ONNX export complete.")


def quantize_model(onnx_input: str, onnx_output: str):
    print("🔧 Quantizing ONNX model to INT8...")
    quantize_dynamic(
        model_input=onnx_input,
        model_output=onnx_output,
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized model saved to: {onnx_output}")


def run_inference(quantized_model_path: str, model_id: str, text: str):
    print("🤖 Running inference on quantized model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

    session = ort.InferenceSession(quantized_model_path)
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    
    # For feature extraction, we get the last hidden state
    last_hidden_state = outputs[0]
    print(f"🧠 Output shape: {last_hidden_state.shape}")
    
    # Apply mean pooling to get sentence embedding (like SentenceTransformer does)
    attention_mask_expanded = np.expand_dims(inputs["attention_mask"], -1)
    attention_mask_expanded = np.broadcast_to(attention_mask_expanded, last_hidden_state.shape)
    
    masked_embeddings = last_hidden_state * attention_mask_expanded
    summed = np.sum(masked_embeddings, axis=1)
    summed_mask = np.sum(attention_mask_expanded, axis=1)
    sentence_embedding = summed / np.maximum(summed_mask, 1e-9)
    
    print(f"🎯 Sentence embedding shape: {sentence_embedding.shape}")
    print(f"🎯 Sentence embedding sample: {sentence_embedding[0][:5]}")


def get_file_size(file_path: str) -> str:
    """Get human-readable file size"""
    if not os.path.exists(file_path):
        return "File not found"
    
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Quantize a SentenceTransformer model to ONNX INT8")
    parser.add_argument("--model", required=True, help="SentenceTransformer model ID (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--output", default="./onnx_model", help="Directory to save ONNX files")
    parser.add_argument("--text", default="I love this product!", help="Test input text for inference")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    onnx_fp32_path = os.path.join(args.output, "model.onnx")
    onnx_int8_path = os.path.join(args.output, "model-quant.onnx")

    try:
        # Export to ONNX
        export_to_onnx(args.model, args.output)
        
        # Check original size
        original_size = get_file_size(onnx_fp32_path)
        print(f"📏 Original ONNX model size: {original_size}")
        
        # Quantize
        quantize_model(onnx_fp32_path, onnx_int8_path)
        
        # Check quantized size
        quantized_size = get_file_size(onnx_int8_path)
        print(f"📏 Quantized ONNX model size: {quantized_size}")
        
        # Test inference
        run_inference(onnx_int8_path, args.model, args.text)
        
        print(f"\n🎉 Success! Quantized model saved to: {onnx_int8_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()