FROM python:3.11-slim-buster

WORKDIR /app

# Install system dependencies for wget
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ONNX model during build
RUN mkdir -p onnx_model && \
    wget -O onnx_model/model-quant.onnx https://huggingface.co/Ndi2020/bhadresh-emotion-onnx/resolve/main/model-quant.onnx

COPY . .

CMD ["gunicorn", "main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:$PORT", "--timeout", "120"]

