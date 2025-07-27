from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from transformers import pipeline
import os
import psutil
import platform
import json
import time
import traceback

app = FastAPI()

MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

# Load model
classifier = pipeline("zero-shot-classification", model=MODEL_NAME)

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

LABELS = ["positive", "negative", "delivery issue", "product quality", "neutral"]

def get_size_in_kb(data: str) -> float:
    return len(data.encode("utf-8")) / 1024

@app.get("/")
def root():
    return {"message": "Zero-shot backend is live."}

@app.post("/predict")
def predict(request: CommentsRequest):
    try:
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        input_json = json.dumps(request.model_dump())
        total_data_size_kb = get_size_in_kb(input_json)

        results = []
        for comment in request.comments:
            result = classifier(comment.body, candidate_labels=LABELS, multi_label=True)
            label_scores = {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}
            top_label = result["labels"][0]

            results.append({
                "id": comment.id,
                "body": comment.body,
                "top_label": top_label,
                "label_scores": label_scores
            })

        current_memory_mb = process.memory_info().rss / 1024 / 1024

        # Peak memory usage (cross-platform)
        if platform.system() == "Windows":
            peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / 1024 / 1024
        elif platform.system() == "Linux":
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        else:
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

        return_payload = {
            "model_used": MODEL_NAME,
            "multi_label": True,
            "candidate_labels": LABELS,
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2),
            "total_data_size_kb": round(total_data_size_kb, 2),
            "total_return_size_kb": round(get_size_in_kb(json.dumps(results)), 2)
        }

        return return_payload

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
