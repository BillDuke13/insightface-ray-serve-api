# InsightFace Ray Serve API

A production-grade REST API for face detection, attribute extraction, and
1:1 comparison, powered by [InspireFace](https://github.com/HyperInspire/InspireFace)
and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html).

## Architecture

Two tiers, one concern each:

- **FaceIngress** (CPU): HTTP, validation, image fetching, preprocessing.
- **FaceInference** (CPU/GPU): native sessions with dynamic request batching,
  called over a Serve handle. Detection uses a full-capability session;
  comparison uses a recognition-only session.

See [docs/api.md](docs/api.md) for endpoints, [docs/baseline.md](docs/baseline.md)
for measured performance, and [docs/migration.md](docs/migration.md) for v1→v2
breaking changes.

## Quickstart

Prerequisites: conda, Python 3.11.

```bash
conda env create -f environment.yml
conda activate insightface-ray-serve-api

serve run serve-cpu.yaml
```

Or with Docker:

```bash
docker build -t faceapi:cpu .
docker run --rm --shm-size=2g -p 8000:8000 faceapi:cpu
```

Smoke test:

```bash
curl localhost:8000/-/readyz
curl localhost:8000/v2/models/insightface:detect \
  -H 'content-type: application/json' \
  -d '{"image": {"type": "url", "url": "https://example.com/a.jpg"}}'
```

### First start downloads models

InspireFace fetches its model pack from ModelScope on first session start;
this needs outbound network access and can take a while on slow links. The
download caches under `~/.inspireface` (in the container:
`/root/.inspireface`) and is reused on later starts. The Docker image's
`HEALTHCHECK` grants a 120s start period for this; raise it with
`--health-start-period` on slow links. A plain `docker run --rm` discards
the cache with the container — mount a volume at `/root/.inspireface`, or
pre-seed a long-lived container with `docker cp` from a warm cache, to avoid
re-downloading on every start.

## Configuration

All behavior knobs are environment variables (see `src/faceapi/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `FACEAPI_MAX_IMAGE_BYTES` | 10485760 | largest accepted image |
| `FACEAPI_FETCH_TIMEOUT_S` / `FACEAPI_CONNECT_TIMEOUT_S` | 10 / 3 | URL fetch timeouts (overall deadline and connect) |
| `FACEAPI_S3_PRESIGN_TTL_S` | 300 | presigned URL lifetime |
| `FACEAPI_ALLOW_PRIVATE_HOSTS` | false | allow non-public URL hosts (tests only) |
| `FACEAPI_DETECTION_THRESHOLD` | 0.5 | detector confidence cutoff |
| `FACEAPI_MIN_FACE_PIXELS` | 1600 | smallest comparable face area |
| `FACEAPI_MAX_FACES` | 50 | cap on faces per image |
| `FACEAPI_MAX_IMAGE_DIMENSION` | 1920 | longest-edge downscale cap |
| `FACEAPI_MAX_IMAGE_PIXELS` | 16777216 | total pixel cap, checked before decode (413 beyond) |
| `FACEAPI_MAX_LANDMARKS` | 106 | landmark points cap |
| `FACEAPI_MAX_INGRESS_CONCURRENCY` | 32 | per-replica guard (429 beyond) |
| `FACEAPI_LOG_LEVEL` | INFO | log verbosity |
| `FACEAPI_OTEL_ENABLED` | false | OTLP tracing (endpoint via `OTEL_*`) |

Scaling and resources live in `serve-cpu.yaml` / `serve-gpu.yaml`.

## Deployment

- CPU: `serve run serve-cpu.yaml` (local) or `serve deploy serve-cpu.yaml`
  (existing cluster).
- GPU: `serve run serve-gpu.yaml` on a GPU cluster. The inference tier takes
  one GPU per replica; InspireFace GPU execution needs the CUDA/TensorRT
  resource bundle on the node (see the InspireFace repo). The GPU YAML is
  schema-validated; GPU execution itself was not run in this environment.
- Probes: liveness `GET /-/healthz` (answered by the Serve proxy),
  readiness `GET /-/readyz` (answered by the app after an inference ping).
- Deployment-level request metrics come from Ray Serve; the app adds
  request IDs, JSON logs with per-stage timings, and optional OTel spans.

## Development

```bash
ruff check src/faceapi tests scripts
ruff format --check src/faceapi tests scripts
mypy src/faceapi tests scripts
pytest
```

CI runs the same gates on every push. Dependencies are pinned in
`pyproject.toml` with a full transitive `requirements.lock`; regenerate it
with `python scripts/freeze.py` after changing the working set.

Version pairing constraint: FastAPI must stay at 0.139.1 — Ray Serve's
ingress rewriter calls `include_router`, and FastAPI ≥ 0.139.2 embeds an
unpicklable lock there, breaking deployment. The pin carries a comment in
`pyproject.toml`; upgrade only after re-verifying `serve run` end to end.

## License

Apache License 2.0. See [LICENSE](LICENSE).
