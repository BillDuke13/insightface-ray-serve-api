"""Runtime configuration loaded from environment variables.

Every operational threshold lives here instead of being hardcoded at call
sites. Serve-level settings (replicas, batch sizes) live in the serve YAML
files; this module covers request handling and inference behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """All tunable knobs for request handling and inference."""

    max_image_bytes: int = 10 * 1024 * 1024
    fetch_timeout_s: float = 10.0
    connect_timeout_s: float = 3.0
    s3_presign_ttl_s: int = 300
    allow_private_hosts: bool = False
    detection_threshold: float = 0.5
    max_faces: int = 50
    min_face_pixels: int = 1600
    max_image_dimension: int = 1920
    max_image_pixels: int = 4096 * 4096
    max_landmarks: int = 106
    max_ingress_concurrency: int = 32
    log_level: str = "INFO"
    otel_enabled: bool = False


def load_settings() -> Settings:
    """Build settings from environment variables with documented defaults."""
    return Settings(
        max_image_bytes=_get_int("FACEAPI_MAX_IMAGE_BYTES", 10 * 1024 * 1024),
        fetch_timeout_s=_get_float("FACEAPI_FETCH_TIMEOUT_S", 10.0),
        connect_timeout_s=_get_float("FACEAPI_CONNECT_TIMEOUT_S", 3.0),
        s3_presign_ttl_s=_get_int("FACEAPI_S3_PRESIGN_TTL_S", 300),
        allow_private_hosts=_get_bool("FACEAPI_ALLOW_PRIVATE_HOSTS", False),
        detection_threshold=_get_float("FACEAPI_DETECTION_THRESHOLD", 0.5),
        max_faces=_get_int("FACEAPI_MAX_FACES", 50),
        min_face_pixels=_get_int("FACEAPI_MIN_FACE_PIXELS", 1600),
        max_image_dimension=_get_int("FACEAPI_MAX_IMAGE_DIMENSION", 1920),
        max_image_pixels=_get_int("FACEAPI_MAX_IMAGE_PIXELS", 4096 * 4096),
        max_landmarks=_get_int("FACEAPI_MAX_LANDMARKS", 106),
        max_ingress_concurrency=_get_int("FACEAPI_MAX_INGRESS_CONCURRENCY", 32),
        log_level=os.environ.get("FACEAPI_LOG_LEVEL", "INFO"),
        otel_enabled=_get_bool("FACEAPI_OTEL_ENABLED", False),
    )


SETTINGS = load_settings()
