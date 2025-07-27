# main.py  (Moritz zero-shot, ONNXRuntime)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import psutil, platform, time, os, json, traceback, requests

MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
ONNX_URL = "https://huggingface.co/Ndi2020/MoritzLaurermultilingual-MiniLMv2-L6-mnli-xnli/resolve/main/model.onnx"
ONNX_PATH = "./onnx_model/model-quant.onnx"
HYPOTHESIS_TEMPLATE = "This text is about {}."
ENTAILMENT_IDX = 2  # [contradiction, neutral, entailment]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def download_if_needed():
    if not os.path.exists(ONNX_PATH):
        os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
        print("⬇️ Downloading ONNX quantized model…")
        with open(ONNX_PATH, "wb") as f:
            f.write(requests.get(ONNX_URL).content)
        print("✅ Download complete.")

download_if_needed()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(ONNX_PATH, so)

app = FastAPI()

class Item(BaseModel):
    id: str
    text: str

class ZeroShotRequest(BaseModel):
    items: List[Item]
    candidate_labels: List[str]
    multi_label: bool = True
    hypothesis_template: str = HYPOTHESIS_TEMPLATE

def kb(s: str) -> float:
    return len(s.encode("utf-8")) / 1024

def get_peak_mb() -> float:
    p = psutil.Process(os.getpid())
    cur = p.memory_info().rss / 1024 / 1024
    if platform.system() == "Linux":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    elif platform.system() == "Windows":
        peak = getattr(p.memory_info(), "peak_wset", cur) / 1024 / 1024
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return max(cur, peak)

@app.get("/")
def root():
    return {"message": "Moritz zero-shot ONNX backend running."}

@app.post("/predict")
def predict(req: ZeroShotRequest):
    try:
        timings = {}
        t0 = time.perf_counter()
        p = psutil.Process(os.getpid())
        mem0 = p.memory_info().rss / 1024 / 1024

        total_data_size_kb = kb(req.model_dump_json())

        results = []
        z0 = time.perf_counter()
        for item in req.items:
            hyps = [req.hypothesis_template.format(lbl) for lbl in req.candidate_labels]

            enc = tokenizer(
                [item.text] * len(hyps),
                hyps,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=512
            )

            inputs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]
            }
            if "token_type_ids" in enc:
                inputs["token_type_ids"] = enc["token_type_ids"]
            else:
                inputs["token_type_ids"] = np.zeros_like(enc["input_ids"])

            t_infer0 = time.perf_counter()
            logits = session.run(None, inputs)[0]  # (num_labels, 3)
            t_infer1 = time.perf_counter()

            # stable softmax
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / np.sum(exp, axis=1, keepdims=True)

            entail = probs[:, ENTAILMENT_IDX]
            label_scores = list(zip(req.candidate_labels, entail.tolist()))
            label_scores.sort(key=lambda x: x[1], reverse=True)

            if req.multi_label:
                top_label, top_score = None, None
            else:
                top_label, top_score = label_scores[0]

            results.append({
                "id": item.id,
                "text": item.text,
                "top_label": top_label,
                "top_score": round(float(top_score), 4) if top_score is not None else None,
                "label_scores": {lbl: round(float(s), 4) for lbl, s in label_scores},
                "forward_time": t_infer1 - t_infer0
            })

        z1 = time.perf_counter()
        timings["zero_shot_total"] = z1 - z0

        mem_peak = get_peak_mb()

        response = {
            "model_used": MODEL_ID,
            "multi_label": req.multi_label,
            "candidate_labels": req.candidate_labels,
            "results": results,
            "memory_initial_mb": round(mem0, 2),
            "memory_peak_mb": round(mem_peak, 2),
        }

        total_return_kb = kb(json.dumps(response))
        response["total_data_size_kb"] = round(total_data_size_kb, 2)
        response["total_return_size_kb"] = round(total_return_kb, 2)

        timings["total_time"] = time.perf_counter() - t0
        response["timing"] = timings

        return response
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
