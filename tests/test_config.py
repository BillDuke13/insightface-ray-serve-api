"""Settings tests."""

from __future__ import annotations

import pytest

from faceapi.config import Settings, load_settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.max_image_bytes == 10 * 1024 * 1024
    assert settings.allow_private_hosts is False
    assert settings.detection_threshold == 0.5
    assert settings.max_image_pixels == 4096 * 4096


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACEAPI_MAX_IMAGE_BYTES", "1024")
    monkeypatch.setenv("FACEAPI_ALLOW_PRIVATE_HOSTS", "true")
    monkeypatch.setenv("FACEAPI_DETECTION_THRESHOLD", "0.7")
    settings = load_settings()
    assert settings.max_image_bytes == 1024
    assert settings.allow_private_hosts is True
    assert settings.detection_threshold == 0.7
