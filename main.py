# for local test
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
MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
ONNX_MODEL_URL = "https://huggingface.co/Ndi2020/j-hartmannemotion-english-distilroberta-base/resolve/main/model-quant.onnx"
ONNX_MODEL_PATH = "./onnx_model/j-hartmann-model-quant.onnx"
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

BATCH_SIZE = 16
THRESHOLD = 0.3
TIMEOUT_SECONDS = 300  # Render hard timeout

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emotion-model")


# === Ensure Model Exists (with retry) ===
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

# === Load Tokenizer and ONNX Session ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

# === FastAPI App ===
app = FastAPI()

process = psutil.Process(os.getpid())

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

# Background task to log memory usage every 10 seconds, add sample memory usage every 1 second
# === Memory monitor globals ===
memory_samples = []
_sample_interval = 0.0001  # seconds
_log_interval = 10  # seconds

async def log_and_sample_memory_usage():
    global memory_samples
    elapsed = 0

    while True:
        mem_mb = process.memory_info().rss / (1024 * 1024)
        memory_samples.append(mem_mb)
        elapsed += _sample_interval

        if elapsed >= _log_interval:
            logging.info(f"[MEMORY MONITOR] Current memory: {mem_mb:.2f} MB")
            elapsed = 0

        await asyncio.sleep(_sample_interval)
        

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_and_sample_memory_usage())

@app.get("/")
def health_check():
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[HEALTH CHECK] Memory usage: {mem_mb:.2f} MB")
    return {
        "status": "backend is alive",
        "message": "j-hartmann ONNX model is running."
    }

@app.get("/warmup")
def warmup():
    logger.info("[WARMUP] Received warmup request")
    return {"status": "warmed"}

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

@app.post("/predict")
async def predict(request: CommentsRequest):
    global memory_samples

    try:
        memory_samples.clear()
        overall_start = time.perf_counter()
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[PREDICT] Initial memory usage: {initial_memory_mb:.2f} MB")
        
        request_json = request.model_dump()
        request_bytes = json.dumps(request_json).encode("utf-8")
        request_size_kb = len(request_bytes) / 1024
        logger.info(f"[PREDICT] Request size: {request_size_kb:.2f} KB")

        texts = [c.body for c in request.comments]
        ids = [c.id for c in request.comments]

        async def stream_results():
            total_response_bytes = 0
            try:
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
                        loop = asyncio.get_running_loop()
                        logits = await asyncio.wait_for(
                            loop.run_in_executor(None, session.run, None, onnx_inputs),
                            timeout=TIMEOUT_SECONDS
                        )
                        logits = logits[0]  # Extract actual logits from output list
                    except asyncio.TimeoutError:
                        logger.error("[PREDICT] Inference call timed out.")
                        raise HTTPException(status_code=504, detail="Inference timed out")

                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

                    for j, p in enumerate(probs):
                        emotion_list = [lbl for idx_, lbl in enumerate(LABELS) if probs[j, idx_] > THRESHOLD]
                        emotion_scores = [{"label": L, "score": round(float(probs[j, idx_]), 4)} for idx_, L in enumerate(LABELS)]


                        result = {
                            "type": "result",
                            "id": batch_ids[j],
                            "emotions": emotion_list,
                            "emotion_scores": emotion_scores
                        }

                        line = json.dumps(result) + "\n"
                        
                        total_response_bytes += len(line.encode("utf-8"))
                        yield line

                    current_memory_mb = process.memory_info().rss / (1024 * 1024)
                    logger.info(f"[PREDICT] Processed batch of {len(batch_texts)} | Memory: {current_memory_mb:.2f} MB")

                    del batch_texts, batch_ids, inputs, onnx_inputs, logits, probs
                    gc.collect()

                # Memory peak stats
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                if platform.system() == "Windows":
                    peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / (1024 * 1024)
                elif platform.system() == "Linux":
                    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                elif platform.system() == "Darwin":
                    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                else:
                    peak_memory_mb = current_memory_mb
                    
                overall_end = time.perf_counter()
                avg_memory = sum(memory_samples) / len(memory_samples)
                gb_seconds = (avg_memory / 1024) * (overall_end - overall_start)

                stats = {
                    "type": "stats",
                    "model_used": MODEL_ID,
                    "memory_initial_mb": round(initial_memory_mb, 2),
                    "memory_peak_mb": round(peak_memory_mb, 2),
                    "modelside_total_memory_gbs": gb_seconds,
                    "total_data_size_kb": round(request_size_kb, 2),
                    "total_return_size_kb": round(total_response_bytes / 1024, 2)
                }

                yield json.dumps(stats) + "\n"

            except asyncio.CancelledError:
                logger.warning("[PREDICT] Streaming task was cancelled — likely due to Render timeout.")
                raise HTTPException(status_code=504, detail="Task was cancelled by host")

        return StreamingResponse(
            stream_results(),
            media_type="application/x-ndjson",
            headers={"Connection": "keep-alive"}
        )

    except Exception as e:
        logger.error("[PREDICT] Exception during prediction", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
