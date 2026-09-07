# CPU image for the face API.
# Build:  docker build -t faceapi:cpu .
# Run:    docker run --rm -p 8000:8000 faceapi:cpu
# GPU: same Dockerfile on an nvidia/cuda:12.x runtime base plus the CUDA and
# TensorRT libraries InspireFace needs (see README); not verified in CI here.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libglib/libgomp: numpy native bits; curl: health probes. No libGL: the app
# uses opencv-python-headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock ./
RUN pip install -r requirements.lock

COPY pyproject.toml serve-cpu.yaml serve-gpu.yaml ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation .

EXPOSE 8000

# First start downloads the InspireFace model pack from ModelScope (needs
# outbound network); the 120s start period below accommodates it. Pre-seed
# /root/.inspireface from a warm cache to skip the download (see README).
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/-/readyz || exit 1

CMD ["serve", "run", "serve-cpu.yaml"]
