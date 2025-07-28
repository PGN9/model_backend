import os
import random
import string
import time
import psutil
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from tqdm import tqdm

# === Settings ===
MODEL_URL = "https://huggingface.co/Ayeshas21/sentence-transformers-all-MiniLM-L6-v2-quantized/resolve/main/model-quant.onnx"
MODEL_FILENAME = "model-quant.onnx"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
NUM_COMMENTS = 100

def download_model():
    print("Step 0/6: Ensure quantized ONNX model is downloaded...")
    if not os.path.exists(MODEL_FILENAME):
        import requests
        print(f"📥 Downloading quantized model from {MODEL_URL}...")
        r = requests.get(MODEL_URL)
        with open(MODEL_FILENAME, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded {MODEL_FILENAME} ({len(r.content)} bytes)")
    else:
        print(f"✅ Model already exists: {MODEL_FILENAME}")

def generate_fake_comments(n):
    print("Step 1/6: Generating fake comments...")
    comments = [
        "This is comment number " + str(i) + ": " + ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 100)))
        for i in range(n)
    ]
    print(f"Generated {len(comments)} comments.")
    return comments

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    download_model()

    comments = generate_fake_comments(NUM_COMMENTS)

    print("Step 2/6: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer loaded.")

    print("Step 3/6: Loading quantized ONNX model...")
    ort_sess = ort.InferenceSession(MODEL_FILENAME, providers=["CPUExecutionProvider"])
    print("Model session initialized.")

    print("Step 4/6: Tokenizing all comments...")
    tokenized = [tokenizer(c, truncation=True, max_length=128) for c in comments]
    print(f"Tokenized {len(tokenized)} comments.")

    num_batches = len(tokenized) // BATCH_SIZE
    embeddings = []
    print(f"Step 5/6: Running inference in {num_batches} batches of {BATCH_SIZE}...")

    mem_start = get_memory_mb()

    for i in range(num_batches):
        print(f"⚙️  Encoding batch {i+1}/{num_batches}...")
        batch = tokenized[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        batch_encoding = tokenizer.pad(batch, padding=True, return_tensors="np")

        # Only include inputs that model accepts
        ort_inputs = {
            k: v for k, v in batch_encoding.items()
            if k in [inp.name for inp in ort_sess.get_inputs()]
        }

        ort_outs = ort_sess.run(None, ort_inputs)
        embeddings.extend(ort_outs[0])  # output shape: (batch_size, embedding_dim)

    mem_end = get_memory_mb()
    print(f"✅ Inference complete. {len(embeddings)} embeddings generated.")
    print(f"📊 RAM usage: start={mem_start:.2f} MB, end={mem_end:.2f} MB, delta={(mem_end - mem_start):.2f} MB")

    print("Step 6/6: Running simple clustering for validation...")

    from sklearn.cluster import KMeans
    K = 5
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(np.array(embeddings))
    print("✅ Clustering complete.")
    print("Cluster centers:", kmeans.cluster_centers_)

if __name__ == "__main__":
    main()
