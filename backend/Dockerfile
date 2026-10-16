FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NV_DATA_DIR=/data
WORKDIR /app

COPY requirements.txt .
# CPU-only PyTorch keeps the image small; use a CUDA base image + CUDA wheels for GPU inference.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY neurovista ./neurovista
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
VOLUME /data

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

EXPOSE 8000
CMD ["uvicorn", "neurovista.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
