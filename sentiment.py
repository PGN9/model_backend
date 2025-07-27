from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import time
import psutil
import os

nli_bp = Blueprint('nli', __name__)

# Global variables to store model and tokenizer
model = None
tokenizer = None
model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_model():
    """Load the quantized model and tokenizer"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print("Loading model and tokenizer...")
        initial_memory = get_memory_usage()
        
        try:
            # Try to load quantized ONNX model first
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = ORTModelForSequenceClassification.from_pretrained(
                model_name,
                export=True,
                provider="CPUExecutionProvider"
            )
            print("Loaded ONNX quantized model")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            # Fallback to regular PyTorch model with dynamic quantization
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Apply dynamic quantization
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("Loaded PyTorch model with dynamic quantization")
        
        peak_memory = get_memory_usage()
        print(f"Memory usage - Initial: {initial_memory:.2f}MB, Peak: {peak_memory:.2f}MB")

@nli_bp.route('/classify', methods=['POST'])
@cross_origin()
def classify_nli():
    """Classify Natural Language Inference relationship between premise and hypothesis"""
    try:
        data = request.get_json()
        
        if not data or 'premise' not in data or 'hypothesis' not in data:
            return jsonify({'error': 'Both premise and hypothesis must be provided'}), 400
        
        premise = data['premise']
        hypothesis = data['hypothesis']
        
        if not premise.strip() or not hypothesis.strip():
            return jsonify({'error': 'Empty premise or hypothesis provided'}), 400
        
        # Load model if not already loaded
        load_model()
        
        start_time = time.time()
        initial_memory = get_memory_usage()
        
        # Format input for NLI model
        input_text = f"{premise} </s> {hypothesis}"
        
        # Tokenize and predict
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get the predicted class
        predicted_class_id = predictions.argmax().item()
        confidence = predictions.max().item()
        
        # Map class ID to label for NLI
        id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        predicted_label = id2label.get(predicted_class_id, 'unknown')
        
        processing_time = time.time() - start_time
        peak_memory = get_memory_usage()
        
        result = {
            'premise': premise,
            'hypothesis': hypothesis,
            'prediction': predicted_label,
            'confidence': float(confidence),
            'all_scores': {
                id2label.get(i, f'class_{i}'): float(score) 
                for i, score in enumerate(predictions[0])
            },
            'metrics': {
                'processing_time': processing_time,
                'memory_initial_mb': initial_memory,
                'memory_peak_mb': peak_memory,
                'model_used': model_name
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@nli_bp.route('/batch_classify', methods=['POST'])
@cross_origin()
def batch_classify_nli():
    """Classify multiple premise-hypothesis pairs"""
    try:
        data = request.get_json()
        
        if not data or 'pairs' not in data:
            return jsonify({'error': 'No pairs provided'}), 400
        
        pairs = data['pairs']
        if not isinstance(pairs, list) or not pairs:
            return jsonify({'error': 'Invalid pairs format or empty list'}), 400
        
        # Load model if not already loaded
        load_model()
        
        start_time = time.time()
        initial_memory = get_memory_usage()
        
        results = []
        
        for pair in pairs:
            if not isinstance(pair, dict) or 'premise' not in pair or 'hypothesis' not in pair:
                results.append({
                    'error': 'Invalid pair format - must contain premise and hypothesis'
                })
                continue
                
            premise = pair['premise']
            hypothesis = pair['hypothesis']
            
            if not premise.strip() or not hypothesis.strip():
                results.append({
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'error': 'Empty premise or hypothesis'
                })
                continue
            
            # Format input for NLI model
            input_text = f"{premise} </s> {hypothesis}"
            
            # Tokenize and predict
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the predicted class
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()
            
            # Map class ID to label for NLI
            id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            predicted_label = id2label.get(predicted_class_id, 'unknown')
            
            results.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'prediction': predicted_label,
                'confidence': float(confidence),
                'all_scores': {
                    id2label.get(i, f'class_{i}'): float(score) 
                    for i, score in enumerate(predictions[0])
                }
            })
        
        processing_time = time.time() - start_time
        peak_memory = get_memory_usage()
        
        response = {
            'results': results,
            'metrics': {
                'total_processing_time': processing_time,
                'memory_initial_mb': initial_memory,
                'memory_peak_mb': peak_memory,
                'model_used': model_name,
                'number_of_pairs': len(pairs)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@nli_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'memory_usage_mb': get_memory_usage()
    })
