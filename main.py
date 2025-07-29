from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.responses import JSONResponse
import os
import time
import psutil
import traceback
import platform
import json
import asyncio

# Lifetime counters
request_count = 0
total_comments = 0
start_time = time.perf_counter()

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

def get_size_in_kb(data: str):
    return len(data.encode("utf-8")) / 1024.0

# Optional memory logger
process = psutil.Process(os.getpid())
async def log_memory():
    while True:
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"[MODEL MEMORY] RSS = {mem_mb:.2f} MB")
        await asyncio.sleep(10)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(log_memory())

@app.get("/")
def root():
    return {"message": "vader backend is running."}

@app.post("/predict")
def predict_sentiment(request: CommentsRequest):
    global request_count, total_comments

    try:
        t0 = time.perf_counter()

        # Memory usage (initial)
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)

        # Payload size (input)
        input_json = json.dumps(request.model_dump())
        total_data_size_kb = get_size_in_kb(input_json)

        # Run sentiment
        results = []
        for comment in request.comments:
            body = comment.body
            scores = analyzer.polarity_scores(body)
            sentiment = (
                "positive" if scores["compound"] > 0.05 else
                "negative" if scores["compound"] < -0.05 else
                "neutral"
            )
            results.append({
                "id": comment.id,
                "body": body,
                "sentiment": sentiment,
                "sentiment_score": scores["compound"]
            })

        # Peak memory usage
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        if platform.system() == "Windows":
            peak = getattr(process.memory_info(), "peak_wset", current_memory_mb * 1024 * 1024)
            peak_memory_mb = peak / (1024 * 1024)
        elif platform.system() == "Linux":
            import resource
            peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_kb / 1024
        elif platform.system() == "Darwin":
            import resource
            peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_bytes / (1024 * 1024)
        else:
            peak_memory_mb = current_memory_mb

        # Total processing time
        t1 = time.perf_counter()
        model_processing_time = round(t1 - t0, 6)

        # Track return size
        return_data = {
            "model_used": "vader",
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2),
            "model_processing_time": model_processing_time,
            "comments_count": len(results),
            "comments_per_second": round(len(results) / model_processing_time, 3) if model_processing_time > 0 else None,
            "uptime_s": round(time.perf_counter() - start_time, 2),
            "requests_total": request_count + 1,
            "comments_total": total_comments + len(results),
        }

        total_return_size_kb = get_size_in_kb(json.dumps(return_data))
        return_data["total_data_size_kb"] = round(total_data_size_kb, 2)
        return_data["total_return_size_kb"] = round(total_return_size_kb, 2)

        # Update counters
        request_count += 1
        total_comments += len(results)

        return return_data

    except Exception as e:
        print("❌ Model error:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8601))
    uvicorn.run("main:app", host="127.0.0.1", port=port)
