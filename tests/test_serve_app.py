"""Serve wiring tests: no cluster needed, decoration runs at import."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator
from typing import Any, ForwardRef, cast, get_args

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from faceapi.errors import LowQualityError, NoFaceError
from faceapi.schemas import ComparisonResponse, DetectionResponse, FaceResult
from faceapi.service import DetectJob, DetectOptions, ImagePair


def _api_routes(app: FastAPI) -> list[APIRoute]:
    def walk(routes: Iterable[Any]) -> Iterator[Any]:
        for route in routes:
            yield route
            yield from walk(getattr(route, "routes", []))
            inner = getattr(route, "original_router", None)
            if inner is not None:
                yield from walk(getattr(inner, "routes", []))

    return [r for r in walk(app.routes) if isinstance(r, APIRoute)]


def _assert_concrete(annotation: Any) -> None:
    assert not isinstance(annotation, (str, ForwardRef)), annotation
    for arg in get_args(annotation):
        _assert_concrete(arg)


def test_route_annotations_are_concrete_types() -> None:
    """Route signatures must survive a pickle round-trip into replicas.

    String annotations (PEP 563) cannot be re-resolved there and degrade
    body models into required query params. See serve_app module docstring.
    """
    from faceapi.serve_app import app

    routes = _api_routes(app)
    assert len(routes) >= 3
    for route in routes:
        signature = inspect.signature(route.endpoint)
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            assert param.annotation is not inspect.Parameter.empty, (route.path, name)
            _assert_concrete(param.annotation)


def test_batch_methods_take_single_list_param() -> None:
    """@serve.batch handlers take exactly one request list (Ray typing)."""
    from faceapi.serve_app import FaceInference

    cls = FaceInference.func_or_class  # type: ignore[attr-defined]
    for method, second in (("detect_batch", "jobs"), ("compare_batch", "pairs")):
        params = list(inspect.signature(getattr(cls, method)).parameters)
        assert params == ["self", second]


def _bare_replica() -> Any:
    """A FaceInference instance without the heavy __init__ (native models)."""
    from faceapi.serve_app import FaceInference

    cls = FaceInference.func_or_class  # type: ignore[attr-defined]
    return cls.__new__(cls)


async def test_detect_batch_isolates_expected_failures() -> None:
    """One bad image must not fail its batch-mates.

    Serve delivers a single raised exception to every request in the batch,
    so the handlers return per-slot results and RayInferenceClient re-raises
    error slots. Tested through the undecorated handler: the @serve.batch
    wrapper's queue binds to a live event loop, unusable under pytest.
    """
    replica = _bare_replica()
    calls: list[Any] = []

    class FakeEngine:
        def detect(
            self,
            image: Any,
            *,
            include_embedding: bool,
            include_landmarks: bool,
            max_faces: int | None,
        ) -> list[Any]:
            calls.append(image)
            if not image.any():
                raise NoFaceError("No face detected in image1.")
            return [FaceResult(bounding_box=(0, 0, 8, 8), confidence=0.9)]

    replica._engine = FakeEngine()
    options = DetectOptions(include_embedding=False, include_landmarks=False, max_faces=None)
    blank = np.zeros((2, 2, 3), dtype=np.uint8)
    good = np.full((2, 2, 3), 7, dtype=np.uint8)
    handler = replica.detect_batch.__wrapped__
    results = await handler(replica, [DetectJob(blank, options), DetectJob(good, options)])
    assert isinstance(results[0], NoFaceError)
    assert isinstance(results[1], DetectionResponse)
    assert results[1].faces[0].confidence == pytest.approx(0.9)
    assert len(calls) == 2  # the first failure did not stop the batch


async def test_compare_batch_isolates_expected_failures() -> None:
    replica = _bare_replica()

    class FakeEngine:
        def compare(self, first: Any, second: Any) -> float:
            if not first.any():
                raise LowQualityError("Primary face in image1 is too small to compare reliably.")
            return 0.5

    replica._engine = FakeEngine()
    blank = np.zeros((2, 2, 3), dtype=np.uint8)
    good = np.full((2, 2, 3), 7, dtype=np.uint8)
    handler = replica.compare_batch.__wrapped__
    results = await handler(replica, [ImagePair(blank, good), ImagePair(good, good)])
    assert isinstance(results[0], LowQualityError)
    assert isinstance(results[1], ComparisonResponse)
    assert results[1].similarity == pytest.approx(0.5)


async def test_inference_client_raises_error_slots() -> None:
    """Error slots returned by batch handlers re-raise per request."""
    from faceapi.serve_app import RayInferenceClient

    class StubMethod:
        def __init__(self, outcome: Any) -> None:
            self._outcome = outcome

        async def remote(self, *args: Any) -> Any:
            return self._outcome

    class StubHandle:
        def __init__(self) -> None:
            self.detect_batch = StubMethod(DetectionResponse(faces=[]))
            self.compare_batch = StubMethod(NoFaceError("No face detected in image2."))
            self.ping = StubMethod("ok")

    client = RayInferenceClient(cast(Any, StubHandle()))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    options = DetectOptions(include_embedding=False, include_landmarks=False, max_faces=None)
    assert (await client.detect(image, options)).faces == []
    with pytest.raises(NoFaceError, match="image2"):
        await client.compare(ImagePair(image, image))
