from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import requests
import os
import json
import gc
import logging
import psutil
import time
import asyncio
import platform
import resource

# === Config ===
MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
ONNX_MODEL_URL = "https://huggingface.co/dakyswr/lxyuan-distilbert-sentiment-onnx/resolve/main/model-quant.onnx"
ONNX_MODEL_PATH = "./onnx_model/model-quant.onnx"
LABELS = ["negative", "neutral", "positive"]
BATCH_SIZE = 8
TIMEOUT_SECONDS = 300

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentiment-model")

# === Download model if not exists ===
def download_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        logger.info("Downloading quantized ONNX model...")
        os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
        for attempt in range(3):
            try:
                response = requests.get(ONNX_MODEL_URL, timeout=60)
                response.raise_for_status()
                with open(ONNX_MODEL_PATH, "wb") as f:
                    f.write(response.content)
                logger.info("Model downloaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/3 failed: {e}")
                time.sleep(2)
        raise RuntimeError("Failed to download ONNX model after 3 attempts.")

download_model()

# === Load model & tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
process = psutil.Process(os.getpid())

# === FastAPI App ===
app = FastAPI()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_memory_usage())

@app.get("/")
def health_check():
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return {"status": "backend is alive", "message": "Sentiment model running."}

@app.get("/metrics")
def get_metrics():
    memory_info = process.memory_info()
    return {
        "memory_usage_mb": round(memory_info.rss / (1024 * 1024), 2),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "num_threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "connections": len(process.connections())
    }

async def log_memory_usage():
    while True:
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[MEMORY MONITOR] Current memory usage: {mem_mb:.2f} MB")
        await asyncio.sleep(10)

@app.post("/predict")
async def predict(request: CommentsRequest):
    try:
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)

        texts = [c.body for c in request.comments]
        ids = [c.id for c in request.comments]
        request_bytes = json.dumps(request.model_dump()).encode("utf-8")
        request_size_kb = len(request_bytes) / 1024

        async def stream_results():
            total_response_bytes = 0
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i+BATCH_SIZE]
                batch_ids = ids[i:i+BATCH_SIZE]

                inputs = tokenizer(batch_texts, return_tensors="np", padding=True, truncation=True, max_length=512)
                onnx_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }

                logits = (await asyncio.to_thread(session.run, None, onnx_inputs))[0]
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                preds = np.argmax(probs, axis=1)

                for j, (pred_idx, prob) in enumerate(zip(preds, probs)):
                    label = LABELS[pred_idx]
                    score = round(float(prob[pred_idx]), 4)
                    result = {
                        "type": "result",
                        "id": batch_ids[j],
                        "sentiment": label,
                        "sentiment_score": score
                    }
                    line = json.dumps(result) + "\n"
                    total_response_bytes += len(line.encode("utf-8"))
                    yield line

                del batch_texts, batch_ids, inputs, onnx_inputs, logits, probs, preds
                gc.collect()

            current_memory_mb = process.memory_info().rss / (1024 * 1024)
            peak_memory_mb = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                (1024 if platform.system() == "Linux" else (1024 * 1024))
            )

            stats = {
                "type": "stats",
                "model_used": MODEL_ID,
                "memory_initial_mb": round(initial_memory_mb, 2),
                "memory_peak_mb": round(peak_memory_mb, 2),
                "total_data_size_kb": round(request_size_kb, 2),
                "total_return_size_kb": round(total_response_bytes / 1024, 2)
            }

            yield json.dumps(stats) + "\n"

        return StreamingResponse(stream_results(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"[PREDICT] Error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
