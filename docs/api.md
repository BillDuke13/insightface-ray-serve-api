# API Reference (v2)

Base URL (local): `http://127.0.0.1:8000`. Interactive docs: `/docs`.

All responses carry `X-Request-ID` (pass your own to trace a call; one is
minted when absent). Errors use a uniform envelope:

```json
{"code": "VALIDATION_ERROR", "message": "...", "request_id": "..."}
```

| Status | Code | Meaning |
|---|---|---|
| 400 | `VALIDATION_ERROR`, `BAD_IMAGE` | malformed request or undecodable image |
| 404 | (proxy) | unknown path |
| 413 | `IMAGE_TOO_LARGE` | image exceeds the byte cap (`FACEAPI_MAX_IMAGE_BYTES`) or the pixel cap (`FACEAPI_MAX_IMAGE_PIXELS`, checked before decode) |
| 422 | `NO_FACE_FOUND`, `LOW_QUALITY_FACE` | no usable face for the operation |
| 429 | `OVERLOADED` | ingress concurrency guard tripped, retry later |
| 502 | `IMAGE_FETCH_FAILED` | the image URL could not be fetched |
| 503 | `NOT_READY` | inference replicas not ready yet |
| 500 | `INTERNAL_ERROR` | unexpected failure, no internals leaked |

## Image sources

Every image field is a discriminated union; exactly one variant is required:

```json
{"type": "base64", "data": "<base64 bytes>"}
{"type": "url", "url": "https://..."}
```

URLs may use `http`, `https`, or `s3`. `s3://bucket/key` inputs are fetched
via a short-lived presigned URL. Hosts that resolve to non-public addresses
are rejected — on the initial URL and on every redirect hop — unless
`FACEAPI_ALLOW_PRIVATE_HOSTS=true` (tests only). Fetches obey an overall
deadline (`FACEAPI_FETCH_TIMEOUT_S`) as well as the byte cap.

## POST /v2/models/insightface:detect

Detect faces and attributes. Faces return highest confidence first.

Request:

```json
{
  "image": {"type": "url", "url": "https://example.com/a.jpg"},
  "include_embedding": false,
  "include_landmarks": true,
  "max_faces": 10
}
```

- `include_embedding` (default `false`): attach the 512-d embedding vector.
- `include_landmarks` (default `true`): attach landmark points (capped at
  `FACEAPI_MAX_LANDMARKS`, overflow flagged by `landmarks_truncated`).
- `max_faces`: per-request cap, clamped by `FACEAPI_MAX_FACES`.

Response:

```json
{
  "faces": [
    {
      "bounding_box": [x1, y1, x2, y2],
      "confidence": 0.99,
      "landmarks": [[x, y]],
      "landmarks_truncated": false,
      "embedding": null,
      "roll": 2.5, "yaw": 1.3, "pitch": 0.7,
      "quality": 0.98,
      "mask_confidence": 0.01,
      "liveness_confidence": 0.96,
      "gender": "Female",
      "age_bracket": "20-29 years old",
      "race": "Asian"
    }
  ]
}
```

No faces is `{"faces": []}` with status 200, not an error.

## POST /v2/models/insightface:compare

Compare the primary (largest) face of two images.

Request:

```json
{
  "image1": {"type": "base64", "data": "<...>"},
  "image2": {"type": "url", "url": "s3://bucket/b.jpg"}
}
```

Response:

```json
{"similarity": 0.85}
```

`similarity` is a cosine value: higher means more similar, and it can be
negative — do not assume a 0–1 range. Suggested starting threshold: `0.47`
(see `docs/baseline.md`); re-calibrate on production data. A missing or
too-small primary face returns 422 instead of a misleading score.

## Probes

- `GET /-/healthz` — answered by the Serve proxy (`success`); use for
  liveness (container `HEALTHCHECK` uses `/-/readyz` instead so traffic only
  starts after models load).
- `GET /-/readyz` — answered by the app (`{"status": "ready"}`) after an
  inference ping; 503 while replicas warm up.

## Examples

```bash
# detect from a URL
curl -s localhost:8000/v2/models/insightface:detect \
  -H 'content-type: application/json' \
  -d '{"image": {"type": "url", "url": "https://example.com/a.jpg"}}'

# compare two local files
A=$(base64 < a.jpg | tr -d '\n'); B=$(base64 < b.jpg | tr -d '\n')
curl -s localhost:8000/v2/models/insightface:compare \
  -H 'content-type: application/json' \
  -d "{\"image1\": {\"type\": \"base64\", \"data\": \"$A\"}, \"image2\": {\"type\": \"base64\", \"data\": \"$B\"}}"
```
