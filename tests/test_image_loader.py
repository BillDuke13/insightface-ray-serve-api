"""Image loader tests using a mocked HTTP transport (no network)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from faceapi.config import Settings
from faceapi.errors import BadImageError, ImageFetchError, ImageTooLargeError
from faceapi.image_loader import ImageLoader
from faceapi.schemas import Base64Image, UrlImage

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _loader(handler: Callable[[httpx.Request], httpx.Response], **overrides: Any) -> ImageLoader:
    settings = Settings(**overrides)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ImageLoader(settings, client=client)


async def test_load_base64() -> None:
    loader = _loader(lambda request: httpx.Response(200, content=b""), allow_private_hosts=True)
    data = await loader.load(Base64Image(data=TINY_PNG_B64))
    assert data[:8] == bytes.fromhex("89504e470d0a1a0a")
    await loader.aclose()


async def test_load_base64_enforces_size_cap() -> None:
    loader = _loader(
        lambda request: httpx.Response(200, content=b""),
        allow_private_hosts=True,
        max_image_bytes=4,
    )
    with pytest.raises(ImageTooLargeError):
        await loader.load(Base64Image(data=TINY_PNG_B64))
    await loader.aclose()


async def test_load_url_streams_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=JPEG_BYTES)

    loader = _loader(handler, allow_private_hosts=True)
    assert await loader.load(UrlImage(url="http://example.com/a.jpg")) == JPEG_BYTES
    await loader.aclose()


async def test_load_url_rejects_early_on_content_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "99999999"}, content=b"x")

    loader = _loader(handler, allow_private_hosts=True, max_image_bytes=8)
    with pytest.raises(ImageTooLargeError):
        await loader.load(UrlImage(url="http://example.com/a.jpg"))
    await loader.aclose()


async def test_load_url_enforces_streaming_cap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"y" * 100)

    loader = _loader(handler, allow_private_hosts=True, max_image_bytes=8)
    with pytest.raises(ImageTooLargeError):
        await loader.load(UrlImage(url="http://example.com/a.jpg"))
    await loader.aclose()


async def test_load_url_rejects_empty_body() -> None:
    loader = _loader(lambda request: httpx.Response(200, content=b""), allow_private_hosts=True)
    with pytest.raises(BadImageError):
        await loader.load(UrlImage(url="http://example.com/a.jpg"))
    await loader.aclose()


async def test_load_url_maps_status_errors() -> None:
    loader = _loader(lambda request: httpx.Response(404, content=b"nope"), allow_private_hosts=True)
    with pytest.raises(ImageFetchError):
        await loader.load(UrlImage(url="http://example.com/a.jpg"))
    await loader.aclose()


@pytest.mark.parametrize(
    "url",
    ["http://localhost/a.jpg", "http://127.0.0.1/a.jpg", "http://169.254.169.254/x"],
)
async def test_load_url_blocks_private_hosts(url: str) -> None:
    loader = _loader(lambda request: httpx.Response(200, content=JPEG_BYTES))
    with pytest.raises(BadImageError):
        await loader.load(UrlImage(url=url))
    await loader.aclose()


async def test_load_url_blocks_private_redirect_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public host redirecting to a link-local IP must be refused mid-hop."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/x"})
        return httpx.Response(200, content=JPEG_BYTES)

    async def fake_is_global(host: str) -> bool:
        return host == "public.example"

    monkeypatch.setattr("faceapi.image_loader._resolve_is_global", fake_is_global)
    loader = _loader(handler)
    # Production clients host-check every hop via the request event hook; the
    # plain mock client injected above lacks it, so mirror the real config.
    loader._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        max_redirects=3,
        event_hooks={"request": [loader._guard_request]},
    )
    loader._owns_client = True
    with pytest.raises(BadImageError):
        await loader.load(UrlImage(url="http://public.example/a.jpg"))
    # Only the initial hop was sent; the redirect target was refused first.
    assert requested == ["http://public.example/a.jpg"]
    await loader.aclose()


async def test_load_url_enforces_overall_deadline() -> None:
    """Slow-drip responses hit the total deadline, not just per-read timeouts."""

    async def slow_drip() -> AsyncIterator[bytes]:
        for _ in range(20):
            yield b"x" * 1024
            await asyncio.sleep(0.02)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=slow_drip())

    loader = _loader(handler, allow_private_hosts=True, fetch_timeout_s=0.05)
    started = time.perf_counter()
    with pytest.raises(ImageFetchError):
        await loader.load(UrlImage(url="http://example.com/a.jpg"))
    assert time.perf_counter() - started < 0.35
    await loader.aclose()


async def test_load_s3_without_key_fails_fast() -> None:
    loader = _loader(lambda request: httpx.Response(200, content=JPEG_BYTES))
    with pytest.raises(BadImageError):
        await loader.load(UrlImage(url="s3://bucket-only"))
    await loader.aclose()


async def test_load_s3_presigns_then_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "presigned.example.com"
        return httpx.Response(200, content=JPEG_BYTES)

    loader = _loader(handler, allow_private_hosts=True)

    class FakeS3:
        def generate_presigned_url(self, op: str, **kwargs: Any) -> str:
            assert op == "get_object"
            assert kwargs["Params"] == {"Bucket": "b", "Key": "k.jpg"}
            return "https://presigned.example.com/k.jpg?sig=1"

    monkeypatch.setattr(loader, "_s3", FakeS3())
    assert await loader.load(UrlImage(url="s3://b/k.jpg")) == JPEG_BYTES
    await loader.aclose()


async def test_load_s3_presign_failure_maps_to_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _loader(lambda request: httpx.Response(200, content=JPEG_BYTES))

    class BrokenS3:
        def generate_presigned_url(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("no credentials")

    monkeypatch.setattr(loader, "_s3", BrokenS3())
    with pytest.raises(ImageFetchError):
        await loader.load(UrlImage(url="s3://b/k.jpg"))
    await loader.aclose()
