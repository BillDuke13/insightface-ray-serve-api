# Performance Baseline

Environment: macOS ARM64 (Mac mini), CPU-only, conda env `insightface-ray-serve-api`
(Python 3.11, Ray 2.58). Probe image: `lena.jpg` 512x512 (1 face). Same probe
shape and concurrency for every row unless noted.

## v1 (legacy `src/api.py`)

| Metric | Value |
|---|---|
| `GET /-/healthz` (proxy) | 200, 4.3 ms |
| detect sequential x20 p50 / p99 / mean | 20.2 / 21.1 / 20.2 ms |
| detect 8-way concurrent x32 p50 / p99 | 137.0 / 194.0 ms |
| detect throughput (8-way) | 55.9 rps |
| compare same-image similarity / latency | 1.0000 / 29.1 ms |

## v2 (new stack, `batch_wait_timeout_s=0.001`)

| Metric | Value | vs v1 |
|---|---|---|
| `GET /-/readyz` (app) | 200, 8.2 ms | — |
| detect sequential x20 p50 / p99 / mean | 19.9 / 39.6 / 20.9 ms | parity (p99 is one outlier in 20 samples) |
| detect 8-way concurrent x32 p50 / p99 | 82.0 / 99.5 ms | p50 −40%, p99 −49% |
| detect throughput (8-way) | 93.4 rps | +67% |
| compare same-image similarity / latency | 1.0000 / 32.0 ms | +10% (extra handle hop) |

Tuning note: with `batch_wait_timeout_s=0.005`, sequential p50 was 24.6 ms —
the idle batch window added ~4 ms. At 0.001 the window cost disappears while
co-arriving requests still batch (concurrency numbers above).

## Similarity calibration (Olivetti faces, 40+40 pairs, indicative)

| Pairs | n | min | p50 | mean | max |
|---|---|---|---|---|---|
| same person | 40 | 0.50 | 0.88 | 0.84 | 0.96 |
| different person | 40 | −0.16 | 0.06 | 0.07 | 0.27 |

Suggested starting threshold: **0.47** (midpoint of medians; zero errors on
this micro-set). Similarities are cosine values and can be negative — callers
must not assume a 0–1 range. Re-calibrate on production data before relying
on the threshold.

## Autoscaling (CPU config, sustained 8-way load)

`FaceInference` scaled 1 → 3 RUNNING replicas (~40 s reaction, default
`upscale_delay_s`). Downscale was not waited out (`downscale_delay_s`
defaults to 600 s).
