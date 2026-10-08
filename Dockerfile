# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder - installs dependencies and generates the model artifact
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Isolated virtual environment that is copied wholesale into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Train the model inside the builder so the artifact always matches the
# scikit-learn version installed in the image (avoids pickle incompatibility).
COPY model/train_model.py model/train_model.py
RUN MODEL_PATH=/build/model/sentiment_model.pkl python model/train_model.py

# ---------------------------------------------------------------------------
# Stage 2: runtime - minimal image with only what is needed to serve requests
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/model/sentiment_model.pkl \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

WORKDIR /app

# Non-root user for a smaller attack surface
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/model /app/model
COPY app/ /app/app/
COPY tests/ /app/tests/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('UVICORN_PORT','8000'), timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8000}"]
