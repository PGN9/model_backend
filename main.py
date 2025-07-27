from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import os
import psutil
import platform
import time
import json
import traceback
import requests

MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
ONNX_URL = "https://huggingface.co/Ndi2020/j-hartmannemotion-english-distilroberta-base/resolve/main/model-quant.onnx"  # Change to your actual ONNX file URL
ONNX_PATH = "./onnx_model/model-quant.onnx"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Label mapping for the model
id2label = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

def download_if_needed():
    if not os.path.exists(ONNX_PATH):
        os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
        print("⬇️ Downloading ONNX model…")
        with open(ONNX_PATH, "wb") as f:
            f.write(requests.get(ONNX_URL).content)
        print("✅ Download complete.")

download_if_needed()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(ONNX_PATH, so)

app = FastAPI()

class Item(BaseModel):
    id: str
    text: str

class EmotionRequest(BaseModel):
    items: List[Item]

def kb(s: str) -> float:
    return len(s.encode("utf-8")) / 1024

def get_peak_mb() -> float:
    p = psutil.Process(os.getpid())
    cur = p.memory_info().rss / 1024 / 1024
    if platform.system() == "Linux":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    elif platform.system() == "Windows":
        peak = getattr(p.memory_info(), "peak_wset", cur) / 1024 / 1024
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return max(cur, peak)

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "Emotion classification ONNX backend running."}


@app.post("/predict")
def predict(req: EmotionRequest):
    try:
        timings = {}
        t0 = time.perf_counter()
        p = psutil.Process(os.getpid())
        mem0 = p.memory_info().rss / 1024 / 1024

        total_data_size_kb = kb(req.model_dump_json())

        results = []
        for item in req.items:
            enc = tokenizer(
                item.text,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=512
            )

            inputs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]
            }
            if "token_type_ids" in enc:
                inputs["token_type_ids"] = enc["token_type_ids"]
            else:
                inputs["token_type_ids"] = np.zeros_like(enc["input_ids"])

            t_infer0 = time.perf_counter()
            logits = session.run(None, inputs)[0][0]  # shape: (num_labels,)
            t_infer1 = time.perf_counter()

            # stable softmax
            logits = logits - np.max(logits)
            exp = np.exp(logits)
            probs = exp / np.sum(exp)

            top_idx = int(np.argmax(probs))
            top_label = id2label[top_idx]
            top_score = float(probs[top_idx])

            label_scores = {id2label[i]: round(float(prob), 4) for i, prob in enumerate(probs)}

            results.append({
                "id": item.id,
                "text": item.text,
                "top_label": top_label,
                "top_score": round(top_score, 4),
                "label_scores": label_scores,
                "forward_time": t_infer1 - t_infer0
            })

        mem_peak = get_peak_mb()

        response = {
            "model_used": MODEL_ID,
            "results": results,
            "memory_initial_mb": round(mem0, 2),
            "memory_peak_mb": round(mem_peak, 2),
            "total_data_size_kb": round(total_data_size_kb, 2),
        }

        timings["total_time"] = time.perf_counter() - t0
        response["timing"] = timings

        return response
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
