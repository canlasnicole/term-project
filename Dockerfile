# One image for the whole project (for the Docker image submission).
#   default command      -> FastAPI backend on port 7860
#   docker-compose.yml   -> runs the same image twice: backend + Streamlit frontend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    YOLO_CONFIG_DIR=/app/.cache/ultralytics \
    DATA_DIR=/app/backend/data \
    PORT=7860

# Small system libraries that OpenCV / Ultralytics need
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a normal user, not root (safer)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install packages (requirements.txt points pip to the CPU-only PyTorch build)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download model weights now, so the first request in the demo is fast
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.DEFAULT)" \
 && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

RUN mkdir -p /app/backend/data && chown -R user:user /app /home/user
USER user

EXPOSE 7860 8501
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
