# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY caoliu/ ./caoliu/
COPY docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh && mkdir -p /data/media

WORKDIR /app/caoliu
ENTRYPOINT ["/app/docker-entrypoint.sh"]
