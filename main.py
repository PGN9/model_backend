from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
import os
import time
import psutil
import traceback
import platform
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import requests
warnings.filterwarnings('ignore')

request_count = 0
start_time = time.time()

app = FastAPI()

# Model loading with fallback options
MODEL_TYPE = "unknown"
model = None
onnx_session = None
tokenizer = None

def download_quantized_model():
    """Download the quantized model from your Hugging Face repo"""
    model_name = "Ayeshas21/sentence-transformers-all-MiniLM-L6-v2-quantized"
    filename = "model-quant.onnx"
    
    if os.path.exists(filename):
        print(f"✅ {filename} already exists, skipping download")
        return filename
    
    try:
        url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
        print(f"📥 Downloading quantized model from {url}...")
        
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded {filename} ({len(response.content)} bytes)")
        return filename
    except Exception as e:
        print(f"❌ Failed to download quantized model: {e}")
        return None

def load_model():
    """Load the best available model with fallback options"""
    global model, onnx_session, tokenizer, MODEL_TYPE
    
    # Option 1: Try to load quantized ONNX model
    try:
        model_path = download_quantized_model()
        if model_path and os.path.exists(model_path):
            import onnxruntime as ort
            from transformers import AutoTokenizer
            
            print("🔄 Loading quantized ONNX model...")
            onnx_session = ort.InferenceSession(model_path)
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            MODEL_TYPE = "quantized-onnx"
            print("✅ Quantized ONNX model loaded successfully!")
            return
    except ImportError:
        print("⚠️ ONNX Runtime not available, trying alternatives...")
    except Exception as e:
        print(f"⚠️ Failed to load quantized model: {e}")
    
    # Option 2: Try smaller SentenceTransformer models
    smaller_models = [
        "sentence-transformers/paraphrase-MiniLM-L3-v2",  # 17MB
        "sentence-transformers/paraphrase-MiniLM-L6-v2",  # 23MB
        "sentence-transformers/all-MiniLM-L12-v2",        # 33MB
        "sentence-transformers/all-MiniLM-L6-v2"          # 90MB (original)
    ]
    
    for model_name in smaller_models:
        try:
            print(f"🔄 Trying to load {model_name}...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            MODEL_TYPE = f"sentence-transformer-{model_name.split('/')[-1]}"
            print(f"✅ Loaded {model_name} successfully!")
            return
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}: {e}")
            continue
    
    raise RuntimeError("❌ Failed to load any model!")

# Load model on startup
try:
    load_model()
except Exception as e:
    print(f"❌ Critical error loading model: {e}")
    # You might want to exit here or use a very basic fallback

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]
    n_clusters: Optional[int] = None

def get_size_in_kb(data):
    return len(data.encode('utf-8')) / 1024

def encode_texts_quantized(texts: List[str]) -> np.ndarray:
    """Encode texts using quantized ONNX model"""
    all_embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        
        outputs = onnx_session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
        
        # Apply mean pooling
        last_hidden_state = outputs[0]
        attention_mask_expanded = np.expand_dims(inputs["attention_mask"], -1)
        attention_mask_expanded = np.broadcast_to(attention_mask_expanded, last_hidden_state.shape)
        
        masked_embeddings = last_hidden_state * attention_mask_expanded
        summed = np.sum(masked_embeddings, axis=1)
        summed_mask = np.sum(attention_mask_expanded, axis=1)
        embedding = summed / np.maximum(summed_mask, 1e-9)
        
        all_embeddings.append(embedding[0])
    
    return np.array(all_embeddings)

def encode_texts(texts: List[str]) -> np.ndarray:
    """Encode texts using the loaded model"""
    if MODEL_TYPE == "quantized-onnx":
        return encode_texts_quantized(texts)
    else:
        return model.encode(texts)

def determine_optimal_clusters(embeddings, max_clusters=10):
    """Determine optimal number of clusters using the elbow method"""
    n_samples = len(embeddings)
    max_clusters = min(max_clusters, n_samples - 1, 10)
    
    if n_samples <= 2:
        return 1
    elif n_samples <= 5:
        return min(2, n_samples - 1)
    
    inertias = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        if k >= n_samples:
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
    
    if len(inertias) < 3:
        return len(inertias)
    
    deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    delta_deltas = [deltas[i] - deltas[i+1] for i in range(len(deltas)-1)]
    
    if delta_deltas:
        elbow_idx = delta_deltas.index(max(delta_deltas)) + 2
        return min(elbow_idx, len(inertias))
    
    return min(3, len(inertias))

def perform_clustering(embeddings, n_clusters=None):
    """Perform K-means clustering on embeddings"""
    if len(embeddings) <= 1:
        return [0], np.array([[0.0]])
    
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(embeddings)
    else:
        n_clusters = min(n_clusters, len(embeddings))
    
    if n_clusters <= 1:
        return [0] * len(embeddings), np.array([[1.0]])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    cluster_centers = kmeans.cluster_centers_
    distances = []
    for i, embedding in enumerate(embeddings):
        cluster_id = cluster_labels[i]
        distance = cosine_similarity([embedding], [cluster_centers[cluster_id]])[0][0]
        distances.append(distance)
    
    return cluster_labels.tolist(), distances

def get_cluster_topics(embeddings, texts, cluster_labels, n_clusters):
    """Generate topic keywords for each cluster"""
    cluster_topics = {}
    
    for cluster_id in range(n_clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
        
        if not cluster_texts:
            cluster_topics[cluster_id] = []
            continue
        
        from collections import Counter
        import re
        
        combined_text = ' '.join(cluster_texts).lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
                     'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
                     'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 
                     'boy', 'did', 'she', 'use', 'say', 'each', 'which', 'their', 'time', 
                     'will', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 
                     'well', 'many', 'some', 'what', 'with', 'have', 'this', 'that', 'they',
                     'been', 'said', 'very', 'were', 'more', 'than', 'also', 'back', 'only',
                     'come', 'work', 'life', 'even', 'right', 'down', 'years', 'think', 'where'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        top_keywords = [word for word, count in word_counts.most_common(5)]
        cluster_topics[cluster_id] = top_keywords
    
    return cluster_topics

@app.get("/")
def root():
    return {
        "message": "Optimized SentenceTransformer clustering backend is running.",
        "model_type": MODEL_TYPE,
        "status": "ready" if (model or onnx_session) else "error"
    }

@app.post("/predict")
def cluster_comments(request: CommentsRequest):
    try:
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024 
        input_json = json.dumps(request.model_dump()) if hasattr(request, "json") else str(request)
        total_data_size_kb = get_size_in_kb(input_json)

        texts = [comment.body for comment in request.comments]
        
        # Generate embeddings
        embeddings = encode_texts(texts)
        
        cluster_labels, similarity_scores = perform_clustering(embeddings, request.n_clusters)
        n_clusters = len(set(cluster_labels))
        
        cluster_topics = get_cluster_topics(embeddings, texts, cluster_labels, n_clusters)
        
        results = []
        for i, comment in enumerate(request.comments):
            cluster_id = cluster_labels[i]
            similarity_score = similarity_scores[i]
            
            results.append({
                "id": comment.id,
                "body": comment.body,
                "cluster": cluster_id,
                "similarity_to_cluster": round(float(similarity_score), 4),
                "cluster_topics": cluster_topics.get(cluster_id, []),
                "embedding_dim": len(embeddings[i])
            })

        current_memory_mb = process.memory_info().rss / 1024 / 1024
        if platform.system() == "Windows":
            peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / 1024 / 1024
        elif platform.system() == "Linux":
            import resource
            peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_kb / 1024
        elif platform.system() == "Darwin":
            import resource
            peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_bytes / 1024 / 1024
        else:
            peak_memory_mb = current_memory_mb

        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_size = cluster_labels.count(cluster_id)
            cluster_stats[cluster_id] = {
                "size": cluster_size,
                "percentage": round((cluster_size / len(results)) * 100, 2),
                "topics": cluster_topics.get(cluster_id, [])
            }

        return_data = {
            "model_used": f"{MODEL_TYPE}",
            "clustering_algorithm": "K-Means",
            "n_clusters": n_clusters,
            "cluster_stats": cluster_stats,
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2)
        }
        
        total_return_size_kb = get_size_in_kb(json.dumps(return_data))
        return_data["total_data_size_kb"] = round(total_data_size_kb, 2)
        return_data["total_return_size_kb"] = round(total_return_size_kb, 2)
        return return_data

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze")
def analyze_comments(request: CommentsRequest):
    return cluster_comments(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)