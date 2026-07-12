# syntax=docker/dockerfile:1
# Multi-stage image: React frontend + preference platform (API / worker)

# ---------- frontend ----------
FROM node:20-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- runtime ----------
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/platform_data/hf_cache \
    TRANSFORMERS_CACHE=/data/platform_data/hf_cache \
    PLATFORM_DATA_DIR=/data/platform_data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Default: CPU PyTorch wheels (document GPU override separately)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Application code (training CLI optional but useful in container)
COPY config.py predict.py dataset.py pack_candidate.py train.py ./
COPY models/ ./models/
COPY utils/ ./utils/
COPY platform_app/ ./platform_app/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Built SPA (FastAPI serves frontend/dist)
COPY --from=frontend-build /frontend/dist ./frontend/dist

RUN mkdir -p /data/platform_data /data/models /data/media

EXPOSE 8080

# Default command is API; compose overrides for worker
CMD ["python", "-m", "uvicorn", "platform_app.main:app", "--host", "0.0.0.0", "--port", "8080"]
