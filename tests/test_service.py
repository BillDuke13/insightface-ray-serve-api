"""Service orchestration tests with a fake inference client."""

from __future__ import annotations

import io
from typing import Any

import httpx
import numpy.typing as npt
import pytest
from PIL import Image

from faceapi.config import Settings
from faceapi.errors import OverloadedError, UnavailableError
from faceapi.image_loader import ImageLoader
from faceapi.schemas import (
    CompareRequest,
    ComparisonResponse,
    DetectionResponse,
    DetectRequest,
    FaceResult,
)
from faceapi.service import DetectOptions, FaceService, ImagePair


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 200, 60)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeInference:
    def __init__(self, faces: int = 1, ping_ok: bool = True) -> None:
        self._faces = faces
        self._ping_ok = ping_ok
        self.detect_calls = 0
        self.compare_calls = 0

    async def detect(self, image: npt.NDArray[Any], options: DetectOptions) -> DetectionResponse:
        self.detect_calls += 1
        assert image.ndim == 3
        faces = [
            FaceResult(
                bounding_box=(0, 0, 8, 8),
                confidence=0.9,
                landmarks=[(1, 1)] if options.include_landmarks else None,
                embedding=[0.1] if options.include_embedding else None,
            )
            for _ in range(self._faces)
        ]
        if options.max_faces is not None:
            faces = faces[: options.max_faces]
        return DetectionResponse(faces=faces)

    async def compare(self, pair: ImagePair) -> ComparisonResponse:
        self.compare_calls += 1
        assert pair.first.shape == pair.second.shape
        return ComparisonResponse(similarity=0.88)

    async def ping(self) -> None:
        if not self._ping_ok:
            raise RuntimeError("down")


def _service(**overrides: Any) -> tuple[FaceService, FakeInference]:
    settings = Settings(allow_private_hosts=True, **overrides)
    payload = _jpeg()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    loader = ImageLoader(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    fake = FakeInference()
    return FaceService(settings, loader, fake), fake


async def test_detect_end_to_end() -> None:
    service, fake = _service()
    response = await service.detect(
        DetectRequest.model_validate({"image": {"type": "url", "url": "http://example.com/a.jpg"}})
    )
    assert len(response.faces) == 1
    assert fake.detect_calls == 1


async def test_detect_forwards_options() -> None:
    service, _ = _service()
    response = await service.detect(
        DetectRequest.model_validate(
            {
                "image": {"type": "url", "url": "http://example.com/a.jpg"},
                "include_embedding": True,
                "include_landmarks": False,
            }
        )
    )
    (face,) = response.faces
    assert face.embedding == [0.1]
    assert face.landmarks is None


async def test_compare_fetches_both_images() -> None:
    service, fake = _service()
    response = await service.compare(
        CompareRequest.model_validate(
            {
                "image1": {"type": "url", "url": "http://example.com/a.jpg"},
                "image2": {"type": "url", "url": "http://example.com/b.jpg"},
            }
        )
    )
    assert response.similarity == pytest.approx(0.88)
    assert fake.compare_calls == 1


async def test_overload_sheds_immediately() -> None:
    service, _ = _service(max_ingress_concurrency=1)
    await service._guard.acquire()
    try:
        with pytest.raises(OverloadedError):
            await service.detect(
                DetectRequest.model_validate(
                    {"image": {"type": "url", "url": "http://example.com/a.jpg"}}
                )
            )
    finally:
        service._guard.release()


async def test_check_ready_ok_and_not_ready() -> None:
    service, _ = _service()
    await service.check_ready()
    bad = FaceService(Settings(), ImageLoader(Settings()), FakeInference(ping_ok=False))
    with pytest.raises(UnavailableError):
        await bad.check_ready(timeout_s=1)
