# Use python:3.12-slim for a minimal, reproducible base image.
# CPU-only PyTorch is installed via the official CPU index URL to keep the
# image size under ~2 GB (vs ~6 GB for CUDA wheels). GPU users should swap
# the base image to a CUDA-enabled pytorch image instead.
# NOTE: the Streamlit dashboard runs PyTorch directly; the old ONNX Runtime
# inference path was removed from dashboard/app.py (2026-09-27).

FROM python:3.12-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller wheel), then the rest of requirements.
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Project source ─────────────────────────────────────────────────────────────
COPY . .

# ── Runtime ───────────────────────────────────────────────────────────────────
# Streamlit default port 8501; override with --server.port at runtime.
EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck: Streamlit serves a /_stcore/health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
