from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline
import os
import psutil
import json
import time
import traceback

app = FastAPI()

# Enable CORS (for Replit requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
)

MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
LABELS = ["positive", "negative", "delivery issue", "product quality", "neutral"]
BATCH_SIZE = 50  # Process 50 comments at a time

# Load model (auto GPU/CPU)
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device_map="auto",  # Optimize for GPU if available
    batch_size=16,     # Faster inference with batches
)

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

def get_size_in_kb(data: str) -> float:
    """Calculate size of JSON data in KB."""
    return len(data.encode("utf-8")) / 1024

@app.get("/")
def health_check():
    return {"status": "ready", "model": MODEL_NAME}

@app.post("/predict")
async def predict(request: CommentsRequest):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    input_size_kb = get_size_in_kb(json.dumps(request.model_dump()))

    results = []
    try:
        # Process in batches to avoid OOM errors
        for i in range(0, len(request.comments), BATCH_SIZE):
            batch = request.comments[i:i + BATCH_SIZE]
            batch_texts = [comment.body for comment in batch]

            # Batch inference
            batch_results = classifier(
                batch_texts,
                candidate_labels=LABELS,
                multi_label=True
            )

            # Format results
            for comment, result in zip(batch, batch_results):
                label_scores = {
                    label: round(score, 4) 
                    for label, score in zip(result["labels"], result["scores"])
                }
                results.append({
                    "id": comment.id,
                    "body": comment.body,
                    "top_label": result["labels"][0],
                    "label_scores": label_scores
                })

        # Memory metrics
        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        return_size_kb = get_size_in_kb(json.dumps(results))
        processing_time = time.time() - start_time

        # Response format (matches your goal)
        return {
            "model_metrics": {
                "model_used": MODEL_NAME,
                "memory_initial_mb": round(initial_memory_mb, 2),
                "memory_peak_mb": round(peak_memory_mb, 2),
                "total_data_size_kb": round(input_size_kb, 2),
                "total_return_size_kb": round(return_size_kb, 2),
            },
            "number_of_comments": len(request.comments),
            "number_updated": len(results),
            "timing": {
                "model_processing_time": round(processing_time, 2),
            },
            "results": results  # Optional: Include if Replit needs labels
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)  # 1 worker for Render free tier
