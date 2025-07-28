import os
import time
import json
import sys
from typing import List, Dict, Any, Tuple

import psutil
import torch
from fastapi import FastAPI, Request
from transformers import pipeline

# -----------------------
# Config
# -----------------------
MODEL_ID = os.getenv("MODEL_ID", "j-hartmann/emotion-english-distilroberta-base")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # pipeline will still auto-batch, this is for chunking
RETURN_ALL_SCORES = True

app = FastAPI(title="Emotion Backend (j-hartmann)")

# -----------------------
# Load model once
# -----------------------
device = 0 if torch.cuda.is_available() else -1
clf = pipeline(
    "text-classification",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    return_all_scores=RETURN_ALL_SCORES,
    device=device,
)


# -----------------------
# Helpers
# -----------------------
def chunk(lst: List[Any], n: int) -> List[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_batch(texts: List[str]) -> List[List[Dict[str, float]]]:
    """
    Run the HF pipeline on a batch of texts.
    Output: list (len=texts) of list of dicts: [{"label": "...", "score": ...}, ...]
    """
    return clf(texts)  # pipeline handles batch inference when list is passed


def summarize_result(id_: Any, label_scores: List[Dict[str, float]]) -> Dict[str, Any]:
    # Sort by score desc to get top "sentiment"
    sorted_scores = sorted(label_scores, key=lambda x: x["score"], reverse=True)
    top = sorted_scores[0]
    return {
        "id": id_,
        "sentiment": top["label"],
        "sentiment_score": float(top["score"]),
        "emotions": [ls["label"] for ls in sorted_scores],
        "emotion_scores": [float(ls["score"]) for ls in sorted_scores],
    }


def memory_mb(proc: psutil.Process) -> float:
    return round(proc.memory_info().rss / 1024 / 1024, 2)


# -----------------------
# Routes
# -----------------------
@app.get("/")
def root():
    return {"status": "ok", "model_id": MODEL_ID}


@app.post("/predict")
async def predict(request: Request):
    """
    Accepts either:
    {
      "comments": [{"id": "...", "body": "text"}, ...]
    }
    or (for compatibility):
    {
      "items": [{"id": "...", "text": "text"}, ...]
    }
    Returns:
    {
      "results": [
        {
          "id": ...,
          "sentiment": "...",
          "sentiment_score": ...,
          "emotions": [...],
          "emotion_scores": [...]
        },
        ...
      ],
      "processing_time_sec": ...,
      "memory_initial_mb": ...,
      "memory_peak_mb": ... (best-effort),
      "total_data_size_kb": ...,
      "total_return_size_kb": ...
    }
    """
    payload = await request.json()

    # Try comments first (what your Replit proxy is using)
    entries = payload.get("comments")
    extractor = "body"
    if entries is None:
        # Fallback to items
        entries = payload.get("items", [])
        extractor = "text"

    if not entries:
        return {"results": [], "error": "No comments/items provided."}

    proc = psutil.Process()
    mem0 = memory_mb(proc)
    t0 = time.perf_counter()

    # Extract ids + texts
    ids: List[Any] = []
    texts: List[str] = []
    for e in entries:
        if "id" not in e or extractor not in e:
            continue
        ids.append(e["id"])
        texts.append(e[extractor])

    all_results: List[Dict[str, Any]] = []

    # Run in batches (optional; pipeline can also take whole list if RAM allows)
    for text_batch, id_batch in zip(chunk(texts, BATCH_SIZE), chunk(ids, BATCH_SIZE)):
        batch_out = process_batch(text_batch)
        # batch_out is List[List[{"label", "score"}]] aligned with text_batch
        for _id, label_scores in zip(id_batch, batch_out):
            all_results.append(summarize_result(_id, label_scores))

    t1 = time.perf_counter()
    mem1 = memory_mb(proc)

    # Render doesn't make it easy to get "peak" memory portably; we'll just report current as peak fallback.
    resp = {
        "results": all_results,
        "processing_time_sec": round(t1 - t0, 3),
        "memory_initial_mb": mem0,
        "memory_peak_mb": max(mem0, mem1),
        "total_data_size_kb": round(sys.getsizeof(json.dumps(payload)) / 1024, 2),
        "total_return_size_kb": round(sys.getsizeof(json.dumps(all_results)) / 1024, 2),
    }
    return resp
