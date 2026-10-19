FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

# glib is needed by OpenCV even in the headless build
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

# Run as a non-root user; /models holds the downloaded Hugging Face models
RUN useradd --create-home appuser && mkdir -p /models && chown -R appuser /app /models
USER appuser

EXPOSE 8000

# Models load on first start, so give the health check time
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health', timeout=4)"

# PORT lets the same image run on platforms that assign a port (e.g. 7860 on Hugging Face)
CMD ["sh", "-c", "uvicorn main:socket_app --host 0.0.0.0 --port ${PORT:-8000}"]
