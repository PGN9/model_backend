import argparse
import os
import numpy as np
from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

def export_to_onnx(model_id: str, output_path: str):
    """Export the DistilBART model to ONNX format"""
    print(f"🔄 Exporting model '{model_id}' to ONNX...")
    
    # For zero-shot classification, we need to specify the right task
    main_export(
        model_name_or_path=model_id,
        output=output_path,
        task="zero-shot-classification",  # Changed from text-classification
        framework="pt"
    )
    print("✅ ONNX export complete.")

def quantize_model(onnx_input: str, onnx_output: str):
    """Quantize the ONNX model to INT8 for smaller size and faster inference"""
    print("🔧 Quantizing ONNX model to INT8...")
    
    quantize_dynamic(
        model_input=onnx_input,
        model_output=onnx_output,
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized model saved to: {onnx_output}")

def test_quantized_model(quantized_model_path: str, model_id: str):
    """Test the quantized model with zero-shot classification"""
    print("🤖 Testing quantized model with zero-shot classification...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Test data
    premise = "I love playing basketball and watching NBA games."
    hypothesis_template = "This text is about {}."
    labels = ["sports", "technology", "food", "politics"]
    
    # Create session
    session = ort.InferenceSession(quantized_model_path)
    
    print(f"📝 Test text: {premise}")
    print(f"🏷️  Testing labels: {labels}")
    
    results = []
    
    for label in labels:
        hypothesis = hypothesis_template.format(label)
        
        # Tokenize the premise and hypothesis pair
        inputs = tokenizer(
            premise, 
            hypothesis, 
            return_tensors="np", 
            padding=True, 
            truncation=True
        )
        
        # Run inference
        outputs = session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
        
        # Get the logits and apply softmax
        logits = outputs[0][0]  # Shape: [3] for [contradiction, neutral, entailment]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        # For zero-shot classification, we typically use the "entailment" score
        entailment_score = probabilities[2]  # Index 2 is usually entailment
        
        results.append((label, entailment_score))
        print(f"   {label}: {entailment_score:.4f}")
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n🎯 Top prediction: {results[0][0]} (score: {results[0][1]:.4f})")

def get_model_size(file_path: str):
    """Get model file size in MB"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

def main():
    parser = argparse.ArgumentParser(description="Quantize DistilBART-MNLI model to ONNX INT8")
    parser.add_argument(
        "--model", 
        default="valhalla/distilbart-mnli-12-1",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output", 
        default="./quantized_distilbart", 
        help="Directory to save ONNX files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # File paths
    onnx_fp32_path = os.path.join(args.output, "model.onnx")
    onnx_int8_path = os.path.join(args.output, "model-quant.onnx")
    
    print("🚀 Starting DistilBART Quantization Process")
    print("=" * 50)
    
    try:
        # Step 1: Export to ONNX
        export_to_onnx(args.model, args.output)
        
        # Check original size
        original_size = get_model_size(onnx_fp32_path)
        print(f"📊 Original ONNX model size: {original_size:.1f} MB")
        
        # Step 2: Quantize
        quantize_model(onnx_fp32_path, onnx_int8_path)
        
        # Check quantized size
        quantized_size = get_model_size(onnx_int8_path)
        print(f"📊 Quantized model size: {quantized_size:.1f} MB")
        
        if original_size > 0:
            reduction = ((original_size - quantized_size) / original_size) * 100
            print(f"🎉 Size reduction: {reduction:.1f}%")
        
        # Step 3: Test the quantized model
        test_quantized_model(onnx_int8_path, args.model)
        
        print("\n" + "=" * 50)
        print("✅ Quantization complete! Files saved in:", args.output)
        print("📁 Files created:")
        print(f"   - model.onnx (original): {original_size:.1f} MB")
        print(f"   - model-quant.onnx (quantized): {quantized_size:.1f} MB")
        print(f"   - tokenizer files")
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()