"""Ray Serve wiring: inference deployment plus FastAPI ingress deployment.

Scaling, resources, and batching defaults live here; production overrides
live in ``serve-cpu.yaml`` / ``serve-gpu.yaml``. Deploy with::

    serve run serve-cpu.yaml     # or serve-gpu.yaml on GPU nodes

NOTE: this module must NOT use ``from __future__ import annotations``.
Route-handler annotations are re-resolved inside serving replicas after a
pickle round-trip, where string annotations become unresolvable ForwardRefs
and collapse body params into required query params.
"""

import asyncio
from typing import Any

import numpy.typing as npt
from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle

from faceapi.config import SETTINGS
from faceapi.engine import FaceEngine
from faceapi.errors import FaceAPIError, register_exception_handlers
from faceapi.image_loader import ImageLoader
from faceapi.observability import RequestIdMiddleware, setup_logging
from faceapi.schemas import (
    CompareRequest,
    ComparisonResponse,
    DetectionResponse,
    DetectRequest,
)
from faceapi.service import DetectJob, DetectOptions, FaceService, ImagePair
from faceapi.telemetry import init_telemetry

app = FastAPI(
    title="Face Analysis API",
    description="Face detection, attributes, and 1:1 comparison.",
    version="2.0.0",
)
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)


@serve.deployment(name="FaceInference", num_replicas=1)
class FaceInference:
    """GPU-bound tier: native sessions and batched model execution."""

    def __init__(self) -> None:
        setup_logging(SETTINGS.log_level)
        init_telemetry(SETTINGS)
        self._engine = FaceEngine(SETTINGS)

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.001)
    async def detect_batch(self, jobs: list[DetectJob]) -> list[DetectionResponse | FaceAPIError]:
        """Detect faces for a batch of images, one entry per image.

        Expected failures are returned in their slot instead of raised:
        Serve delivers a single raised exception to *every* request in the
        batch, so one bad image must not fail its batch-mates.
        :class:`RayInferenceClient` re-raises error slots. Unexpected
        failures still propagate and fail the whole batch. Model calls run
        in worker threads so the replica event loop stays free for health
        checks.
        """
        results: list[DetectionResponse | FaceAPIError] = []
        for job in jobs:
            try:
                faces = await asyncio.to_thread(
                    self._engine.detect,
                    job.image,
                    include_embedding=job.options.include_embedding,
                    include_landmarks=job.options.include_landmarks,
                    max_faces=job.options.max_faces,
                )
                results.append(DetectionResponse(faces=faces))
            except FaceAPIError as exc:
                results.append(exc)
        return results

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.001)
    async def compare_batch(
        self, pairs: list[ImagePair]
    ) -> list[ComparisonResponse | FaceAPIError]:
        """Compare one entry per image pair; see detect_batch for error slots."""
        results: list[ComparisonResponse | FaceAPIError] = []
        for pair in pairs:
            try:
                similarity = await asyncio.to_thread(self._engine.compare, pair.first, pair.second)
                results.append(ComparisonResponse(similarity=similarity))
            except FaceAPIError as exc:
                results.append(exc)
        return results

    async def ping(self) -> str:
        """Lightweight replica check used by ingress readiness."""
        self._engine.assert_usable()
        return "ok"

    async def check_health(self) -> None:
        """Fail the replica when native sessions are gone."""
        self._engine.assert_usable()


class RayInferenceClient:
    """Adapt a Serve handle to the plain client protocol used by the service."""

    def __init__(self, handle: DeploymentHandle[FaceInference]) -> None:
        self._handle = handle

    async def detect(self, image: npt.NDArray[Any], options: DetectOptions) -> DetectionResponse:
        result: DetectionResponse | FaceAPIError = await self._handle.detect_batch.remote(
            DetectJob(image, options)
        )
        if isinstance(result, FaceAPIError):
            raise result
        return result

    async def compare(self, pair: ImagePair) -> ComparisonResponse:
        result: ComparisonResponse | FaceAPIError = await self._handle.compare_batch.remote(pair)
        if isinstance(result, FaceAPIError):
            raise result
        return result

    async def ping(self) -> None:
        await self._handle.ping.remote()


@serve.deployment(name="FaceIngress", num_replicas=1)
@serve.ingress(app)
class FaceIngress:
    """CPU-bound tier: HTTP, validation, image fetching, and routing."""

    def __init__(self, handle: DeploymentHandle[FaceInference]) -> None:
        setup_logging(SETTINGS.log_level)
        init_telemetry(SETTINGS)
        self._service = FaceService(SETTINGS, ImageLoader(SETTINGS), RayInferenceClient(handle))

    # NOTE: the body param is named `payload`, not `request`: FastAPI reserves
    # the name `request` for the raw Starlette request and misclassifies a
    # same-named body model as a query param.
    @app.post("/v2/models/insightface:detect", response_model=DetectionResponse)
    async def detect(self, payload: DetectRequest) -> DetectionResponse:
        """Detect faces and attributes in one image."""
        return await self._service.detect(payload)

    @app.post("/v2/models/insightface:compare", response_model=ComparisonResponse)
    async def compare(self, payload: CompareRequest) -> ComparisonResponse:
        """Compare the primary faces of two images."""
        return await self._service.compare(payload)

    # NOTE: no /-/healthz route here on purpose. The Serve HTTP proxy owns
    # /-/healthz (it answers "success") and never routes it to the app, so an
    # app-level handler would be dead code. Liveness = proxy /-/healthz,
    # readiness = this /-/readyz below (includes an inference ping).
    @app.get("/-/readyz")
    async def ready(self) -> dict[str, str]:
        """Readiness: inference replicas answer (503 otherwise)."""
        await self._service.check_ready()
        return {"status": "ready"}


# `bind` exists on the Deployment objects produced by @serve.deployment,
# which mypy cannot see through the decorators.
ingress = FaceIngress.bind(FaceInference.bind())  # type: ignore[attr-defined]
