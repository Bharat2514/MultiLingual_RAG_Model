# syntax=docker/dockerfile:1.7
# =============================================================================
# PolyDoc backend — API and ingestion worker share one image.
# Stage `base` carries the system deps (Tesseract Hindi+English per FR-4);
# `api` is the slim serving image; `worker` adds the ML extras.
# =============================================================================

ARG PYTHON_VERSION=3.11

# --- base --------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/data/models \
    TRANSFORMERS_CACHE=/app/data/models

# tesseract-ocr-hin/-eng: FR-4 requires Hindi + English OCR.
# libgl1 + libglib2.0-0: OpenCV runtime for the scan preprocessing pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-hin \
        tesseract-ocr-eng \
        libtesseract-dev \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        fonts-deva \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --system --gid 1001 polydoc \
    && useradd --system --uid 1001 --gid polydoc --create-home polydoc

# --- deps --------------------------------------------------------------------
FROM base AS deps

COPY backend/pyproject.toml /app/backend/pyproject.toml
RUN mkdir -p /app/backend/app && touch /app/backend/app/__init__.py \
    && pip install --upgrade pip \
    && pip install -e /app/backend

# --- ml ----------------------------------------------------------------------
# CPU torch wheels: the container never sees Apple MPS, and CUDA wheels would
# add several GB for nothing. Both api (query embedding + rerank) and worker
# (ingestion embedding) load bge-m3, so both need this layer.
FROM deps AS ml

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && pip install "FlagEmbedding>=1.3" "transformers>=4.45"

# --- api ---------------------------------------------------------------------
FROM ml AS api

COPY --chown=polydoc:polydoc backend /app/backend
RUN mkdir -p /app/data/uploads /app/data/models && chown -R polydoc:polydoc /app/data

USER polydoc
WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

# --- worker ------------------------------------------------------------------
FROM ml AS worker

COPY --chown=polydoc:polydoc backend /app/backend
RUN mkdir -p /app/data/uploads /app/data/models && chown -R polydoc:polydoc /app/data

USER polydoc
WORKDIR /app/backend

CMD ["celery", "-A", "app.workers.celery_app:celery_app", "worker", \
     "--loglevel=INFO", "--concurrency=2", "-Q", "ingestion"]
