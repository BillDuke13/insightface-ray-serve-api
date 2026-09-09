"""Schema validation tests."""

import pytest
from pydantic import TypeAdapter, ValidationError

from faceapi.schemas import (
    Base64Image,
    CompareRequest,
    DetectRequest,
    ImageSource,
    UrlImage,
)

SOURCE: TypeAdapter[ImageSource] = TypeAdapter(ImageSource)

# 1x1 red PNG.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)


def test_base64_source_accepts_valid_payload() -> None:
    source = SOURCE.validate_python({"type": "base64", "data": TINY_PNG_B64})
    assert isinstance(source, Base64Image)


def test_base64_source_tolerates_whitespace() -> None:
    source = SOURCE.validate_python({"type": "base64", "data": f"  {TINY_PNG_B64}\n"})
    assert isinstance(source, Base64Image)


@pytest.mark.parametrize("data", ["", "   ", "not-base64!!!", "a"])
def test_base64_source_rejects_bad_payload(data: str) -> None:
    with pytest.raises(ValidationError):
        SOURCE.validate_python({"type": "base64", "data": data})


@pytest.mark.parametrize(
    "url",
    ["http://example.com/a.jpg", "https://example.com/a.jpg", "s3://bucket/key.jpg"],
)
def test_url_source_accepts_supported_schemes(url: str) -> None:
    source = SOURCE.validate_python({"type": "url", "url": url})
    assert isinstance(source, UrlImage)


@pytest.mark.parametrize("url", ["ftp://example.com/a.jpg", "https://", "notaurl"])
def test_url_source_rejects_bad_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        SOURCE.validate_python({"type": "url", "url": url})


def test_source_requires_discriminator() -> None:
    with pytest.raises(ValidationError):
        SOURCE.validate_python({"data": TINY_PNG_B64})


def test_detect_request_defaults() -> None:
    request = DetectRequest.model_validate(
        {"image": {"type": "url", "url": "https://example.com/a.jpg"}}
    )
    assert request.include_embedding is False
    assert request.include_landmarks is True
    assert request.max_faces is None


@pytest.mark.parametrize("max_faces", [0, 201])
def test_detect_request_bounds_max_faces(max_faces: int) -> None:
    with pytest.raises(ValidationError):
        DetectRequest.model_validate(
            {
                "image": {"type": "url", "url": "https://example.com/a.jpg"},
                "max_faces": max_faces,
            }
        )


def test_compare_request_needs_two_sources() -> None:
    with pytest.raises(ValidationError):
        CompareRequest.model_validate(
            {"image1": {"type": "url", "url": "https://example.com/a.jpg"}}
        )
