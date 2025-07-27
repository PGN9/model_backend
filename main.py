from fastapi import FastAPI
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

# === Config ===
MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
ONNX_MODEL_URL = "https://huggingface.co/Ndi2020/bhadresh-emotion-onnx/resolve/main/model-quant.onnx"
ONNX_MODEL_PATH = "./onnx_model/model-quant.onnx"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
BATCH_SIZE = 32         # Safe for <=512MB RAM
THRESHOLD = 0.3
TIMEOUT_SECONDS = 300   # Render hard timeout is 300s max

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-model")

# === Ensure Model Exists ===
def download_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        logger.info("Downloading quantized ONNX model...")
        os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
        response = requests.get(ONNX_MODEL_URL, timeout=60)
        response.raise_for_status()
        with open(ONNX_MODEL_PATH, "wb") as f:
            f.write(response.content)
        logger.info("Download complete.")

download_model()

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

# === Create Semaphore Globally ===
semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent predictions

# === FastAPI App ===
app = FastAPI()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

@app.get("/")
def health_check():
    return {
        "status": "backend is alive",
        "message": "Emotion ONNX model is running."
    }

@app.get("/metrics")
def get_metrics():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking

    return {
        "memory_usage_mb": round(memory_info.rss / (1024 * 1024), 2),
        "cpu_percent": cpu_percent,
        "num_threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "connections": len(process.connections())
    }

import platform
import resource

@app.post("/predict")
async def predict(request: CommentsRequest):
    async with semaphore:
        try:
            process = psutil.Process(os.getpid())
            initial_memory_mb = process.memory_info().rss / (1024 * 1024)

            request_json = request.model_dump()
            request_bytes = json.dumps(request_json).encode("utf-8")
            request_size_kb = len(request_bytes) / 1024

            texts = [c.body for c in request.comments]
            ids = [c.id for c in request.comments]

            async def stream_results():
                for i in range(0, len(texts), BATCH_SIZE):
                    batch_texts = texts[i:i + BATCH_SIZE]
                    batch_ids = ids[i:i + BATCH_SIZE]

                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="np",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )

                    onnx_inputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }

                    try:
                        # Use asyncio.to_thread to run the blocking ONNX inference in a separate thread
                        logits = (await asyncio.wait_for(
                            asyncio.to_thread(session.run, None, onnx_inputs),
                            timeout=TIMEOUT_SECONDS
                        ))[0]
                    except asyncio.TimeoutError:
                        logger.error("Inference call timed out.")
                        raise HTTPException(status_code=504, detail="Inference timed out")

                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

                    for j, p in enumerate(probs):
                        emotion_list = [label for k, label in enumerate(LABELS) if p[k] > THRESHOLD]
                        emotion_scores = [{"label": label, "score": round(float(p[k]), 4)} for k, label in enumerate(LABELS)]

                        result = {
                            "type": "result",
                            "id": batch_ids[j],
                            "emotions": emotion_list,
                            "emotion_scores": emotion_scores
                        }

                        yield json.dumps(result) + "\n"
                        await asyncio.sleep(0)

                    del batch_texts, batch_ids, inputs, onnx_inputs, logits, probs
                    gc.collect()

                # Calculate peak memory cross-platform
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                if platform.system() == "Windows":
                    peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / (1024 * 1024)
                elif platform.system() == "Linux":
                    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    peak_memory_mb = peak_memory_kb / 1024
                elif platform.system() == "Darwin":
                    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
                else:
                    peak_memory_mb = current_memory_mb

                # Yield final stats
                stats = {
                    "type": "stats",
                    "model_used": MODEL_ID,
                    "memory_initial_mb": round(initial_memory_mb, 2),
                    "memory_peak_mb": round(peak_memory_mb, 2),
                    "total_data_size_kb": round(request_size_kb, 2)
                }
                yield json.dumps(stats) + "\n"

            return StreamingResponse(stream_results(), media_type="application/json")

        except Exception as e:
            logger.error("Exception during prediction", exc_info=True)
            return JSONResponse(status_code=500, content={"error": str(e)})

