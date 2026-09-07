"""Error envelope and handler tests over ASGI (no network)."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request

from faceapi.errors import (
    BadImageError,
    ImageFetchError,
    ImageTooLargeError,
    LowQualityError,
    NoFaceError,
    OverloadedError,
    UnavailableError,
    register_exception_handlers,
)
from faceapi.observability import REQUEST_ID_HEADER, RequestIdMiddleware
from faceapi.schemas import Base64Image


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    @app.get("/boom/{kind}")
    def boom(kind: str) -> None:
        errors = {
            "bad": BadImageError("bad pixels"),
            "large": ImageTooLargeError("too big"),
            "fetch": ImageFetchError("nope"),
            "noface": NoFaceError("empty"),
            "quality": LowQualityError("blurry"),
            "busy": OverloadedError("shed"),
            "down": UnavailableError("warming up"),
            "bug": RuntimeError("kaboom-secret"),
        }
        raise errors[kind]

    @app.post("/echo")
    def echo(img: Base64Image, request: Request) -> dict[str, str]:
        return {"request_id": request.state.request_id}

    return app


@pytest.mark.parametrize(
    ("kind", "status", "code"),
    [
        ("bad", 400, "BAD_IMAGE"),
        ("large", 413, "IMAGE_TOO_LARGE"),
        ("fetch", 502, "IMAGE_FETCH_FAILED"),
        ("noface", 422, "NO_FACE_FOUND"),
        ("quality", 422, "LOW_QUALITY_FACE"),
        ("busy", 429, "OVERLOADED"),
        ("down", 503, "NOT_READY"),
        ("bug", 500, "INTERNAL_ERROR"),
    ],
)
async def test_error_envelope(kind: str, status: int, code: str) -> None:
    transport = httpx.ASGITransport(app=_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.get(f"/boom/{kind}", headers={REQUEST_ID_HEADER: "req-1"})
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert body["request_id"] == "req-1"
    assert response.headers[REQUEST_ID_HEADER] == "req-1"
    if kind == "bug":
        assert "kaboom" not in body["message"]


async def test_validation_error_rejects_huge_payload_without_echo() -> None:
    transport = httpx.ASGITransport(app=_app(), raise_app_exceptions=False)
    payload = "!" * 5000
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.post("/echo", json={"type": "base64", "data": payload})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert payload not in body["message"]
    assert len(body["message"]) <= 320


async def test_request_id_minted_when_missing() -> None:
    transport = httpx.ASGITransport(app=_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.get("/boom/bad")
    assert response.json()["request_id"] not in {"", "-", None}
    assert response.headers[REQUEST_ID_HEADER] == response.json()["request_id"]
