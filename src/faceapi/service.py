"""Ingress orchestration: fetch, decode, and fan out to inference.

:class:`FaceService` owns the whole request path except model execution. It is
a plain class (no Ray dependency) so it can be unit-tested with a fake
:class:`InferenceClient`; the Ray wiring lives in :mod:`faceapi.serve_app`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy.typing as npt

from faceapi.config import Settings
from faceapi.errors import OverloadedError, UnavailableError
from faceapi.image_loader import ImageLoader
from faceapi.preprocess import decode_image
from faceapi.schemas import (
    CompareRequest,
    ComparisonResponse,
    DetectionResponse,
    DetectRequest,
    ImageSource,
)
from faceapi.telemetry import StageTimer, start_span

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectOptions:
    """Per-request detect switches forwarded to the inference tier."""

    include_embedding: bool
    include_landmarks: bool
    max_faces: int | None


@dataclass(frozen=True, slots=True, eq=False)
class DetectJob:
    """One batched detect unit: image plus its options."""

    image: npt.NDArray[Any]
    options: DetectOptions


@dataclass(frozen=True, slots=True, eq=False)
class ImagePair:
    """Two decoded images for comparison (eq disabled: arrays don't compare)."""

    first: npt.NDArray[Any]
    second: npt.NDArray[Any]


class InferenceClient(Protocol):
    """What the service needs from the inference tier."""

    async def detect(
        self, image: npt.NDArray[Any], options: DetectOptions
    ) -> DetectionResponse: ...
    async def compare(self, pair: ImagePair) -> ComparisonResponse: ...
    async def ping(self) -> None: ...


class FaceService:
    """Coordinate one request from image source to API response."""

    def __init__(self, settings: Settings, loader: ImageLoader, inference: InferenceClient) -> None:
        self._settings = settings
        self._loader = loader
        self._inference = inference
        self._guard = asyncio.Semaphore(settings.max_ingress_concurrency)

    def _admit(self) -> None:
        if self._guard.locked():
            raise OverloadedError("Server is overloaded, retry later.")

    async def detect(self, request: DetectRequest) -> DetectionResponse:
        """Run detection: fetch -> decode -> inference."""
        self._admit()
        async with self._guard:
            timer = StageTimer()
            with start_span("fetch_image"):
                data = await self._loader.load(request.image)
            timer.mark("fetch_ms")
            with start_span("decode_image"):
                image = await self._decode_image(data)
            timer.mark("decode_ms")
            options = DetectOptions(
                include_embedding=request.include_embedding,
                include_landmarks=request.include_landmarks,
                max_faces=request.max_faces,
            )
            with start_span("inference.detect"):
                response = await self._inference.detect(image, options)
            timer.mark("inference_ms")
            logger.info(
                "detect done: faces=%d image_bytes=%d %s",
                len(response.faces),
                len(data),
                timer.summary(),
            )
            return response

    async def compare(self, request: CompareRequest) -> ComparisonResponse:
        """Run comparison: fetch both images concurrently, then compare."""
        self._admit()
        async with self._guard:
            timer = StageTimer()
            with start_span("fetch_images"):
                first_bytes, second_bytes = await asyncio.gather(
                    self._fetch_one(request.image1, "image1"),
                    self._fetch_one(request.image2, "image2"),
                )
            timer.mark("fetch_ms")
            with start_span("decode_images"):
                first, second = await asyncio.gather(
                    self._decode_image(first_bytes),
                    self._decode_image(second_bytes),
                )
            timer.mark("decode_ms")
            with start_span("inference.compare"):
                response = await self._inference.compare(ImagePair(first, second))
            timer.mark("inference_ms")
            logger.info("compare done: similarity=%.4f %s", response.similarity, timer.summary())
            return response

    async def _decode_image(self, data: bytes) -> npt.NDArray[Any]:
        """Decode bytes off the event loop with the configured size caps."""
        return await asyncio.to_thread(
            decode_image,
            data,
            max_dimension=self._settings.max_image_dimension,
            max_pixels=self._settings.max_image_pixels,
        )

    async def _fetch_one(self, source: ImageSource, label: str) -> bytes:
        started = time.perf_counter()
        data = await self._loader.load(source)
        logger.debug(
            "fetched %s: bytes=%d elapsed_ms=%.1f",
            label,
            len(data),
            (time.perf_counter() - started) * 1000,
        )
        return data

    async def check_ready(self, timeout_s: float = 5.0) -> None:
        """Verify inference replicas answer; raise 503 when they don't."""
        try:
            await asyncio.wait_for(self._inference.ping(), timeout_s)
        except UnavailableError:
            raise
        except Exception as exc:
            raise UnavailableError("Inference replicas are not ready.") from exc
