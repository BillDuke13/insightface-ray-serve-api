# v1 → v2 Migration Notes

v2 is a clean break: no v1 request or response shape is preserved.

## Endpoint paths

| v1 | v2 |
|---|---|
| `POST /v1/models/insightface:detect` | `POST /v2/models/insightface:detect` |
| `POST /v1/models/insightface:compare` | `POST /v2/models/insightface:compare` |
| `GET /-/healthz` (app, shadowed) | `GET /-/healthz` (proxy `success`) + `GET /-/readyz` (app) |

## Request shapes

v1 used flat `image_base64` / `image_url` fields with an exactly-one rule.
v2 uses a discriminated image source:

```json
// v1
{"image_base64": "<...>"}
// v2
{"image": {"type": "base64", "data": "<...>"}}
```

```json
// v1 compare
{"image1_base64": "<...>", "image2_url": "s3://b/k.jpg"}
// v2 compare
{"image1": {"type": "base64", "data": "<...>"},
 "image2": {"type": "url", "url": "s3://b/k.jpg"}}
```

New detect options: `include_embedding` (default false), `include_landmarks`
(default true), `max_faces`.

## Response shapes

- `FaceResult.feature` is renamed to `embedding` and is `null` unless
  `include_embedding: true`.
- New field `landmarks_truncated` (landmarks are capped).
- Similarity is unchanged in meaning but documented as cosine (may be
  negative); the v1 README's 0–1 claim was wrong.
- Errors: v1 returned FastAPI defaults and leaked internals on 500s. v2
  returns `{code, message, request_id}` for every 4xx/5xx; 500s never leak.

## Behavior changes

- Compare uses the largest face per image (v1 used the first detection) and
  rejects missing/too-small primary faces with 422 instead of scoring noise.
- URL fetching enforces size caps, timeouts, and SSRF host guards; oversize
  images return 413, fetch failures 502.
- Bursts beyond ingress concurrency return 429 (shed fast, retry later).
