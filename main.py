from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from fastapi.responses import JSONResponse
import os
import time
import psutil
import traceback
import platform
import json
import torch

request_count = 0
start_time = time.time()

app = FastAPI()

# Initialize the zero-shot classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Define your topic labels for classification
TOPIC_LABELS = [
    "politics",
    "technology", 
    "sports",
    "entertainment",
    "health",
    "business",
    "science",
    "education",
    "travel",
    "food",
    "relationships",
    "lifestyle",
    "news",
    "opinion",
    "question",
    "complaint",
    "compliment",
    "humor",
    "other"
]

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

def get_size_in_kb(data):
    return len(data.encode('utf-8')) / 1024  # size in KB

@app.get("/")
def root():
    return {"message": "distilbart zero-shot classification backend is running."}

@app.post("/predict")
def predict_topics(request: CommentsRequest):
    try:
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024 
        
        # Track input size
        input_json = json.dumps(request.model_dump()) if hasattr(request, "model_dump") else str(request)
        total_data_size_kb = get_size_in_kb(input_json)

        results = []
        
        for comment in request.comments:
            body = comment.body
            
            # Skip empty or very short comments
            if not body or len(body.strip()) < 3:
                results.append({
                    "id": comment.id,
                    "body": body,
                    "topic": "other",
                    "topic_score": 0.0,
                    "all_topic_scores": {}
                })
                continue
            
            # Truncate very long texts to avoid memory issues
            if len(body) > 1000:
                body = body[:1000] + "..."
            
            # Perform zero-shot classification
            classification_result = classifier(body, TOPIC_LABELS)
            
            # Get the top prediction
            top_topic = classification_result['labels'][0]
            top_score = classification_result['scores'][0]
            
            # Create a dictionary of all topic scores
            topic_scores = dict(zip(classification_result['labels'], classification_result['scores']))
            
            results.append({
                "id": comment.id,
                "body": comment.body,  # Return original body
                "topic": top_topic,
                "topic_score": round(top_score, 4),
                "all_topic_scores": {k: round(v, 4) for k, v in topic_scores.items()}
            })

        # Memory usage check
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Peak memory usage (cross-platform)
        if platform.system() == "Windows":
            peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / 1024 / 1024
        elif platform.system() == "Linux":
            import resource
            peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_kb / 1024
        elif platform.system() == "Darwin":  # macOS
            import resource
            peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_bytes / 1024 / 1024
        else:
            peak_memory_mb = current_memory_mb  # fallback

        return_data = {
            "model_used": "valhalla/distilbart-mnli-12-1",
            "task": "zero-shot-classification",
            "topic_labels": TOPIC_LABELS,
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2),
            "comments_processed": len(results),
            "device_used": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Add data size info
        total_return_size_kb = get_size_in_kb(json.dumps(return_data))
        return_data["total_data_size_kb"] = round(total_data_size_kb, 2)
        return_data["total_return_size_kb"] = round(total_return_size_kb, 2)
        
        return return_data

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "valhalla/distilbart-mnli-12-1",
        "task": "zero-shot-classification",
        "available_topics": TOPIC_LABELS,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)