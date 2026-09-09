"""Async image fetching with size caps and SSRF guards.

All remote images travel the same path: ``s3://`` URLs are converted to
presigned https URLs (signing runs in a worker thread — resolving AWS
credentials can touch the instance metadata service), then fetched with a
shared ``httpx.AsyncClient`` using bounded streaming reads. Every request
hop — the initial URL and each redirect target — is host-checked before it
is sent.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import logging
import socket
from typing import Any, assert_never
from urllib.parse import urlparse

import boto3
import httpx

from faceapi.config import Settings
from faceapi.errors import BadImageError, ImageFetchError, ImageTooLargeError
from faceapi.schemas import Base64Image, ImageSource, UrlImage

logger = logging.getLogger(__name__)


def _looks_private(host: str) -> bool:
    """Best-effort synchronous check for non-routable literal hosts."""
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


async def _resolve_is_global(host: str) -> bool:
    """Resolve a hostname and report whether every address is globally routable."""
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except socket.gaierror:
        return False
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    for raw in addresses:
        try:
            if not ipaddress.ip_address(raw).is_global:
                return False
        except ValueError:
            return False
    return True


class ImageLoader:
    """Fetch raw image bytes from any supported image source."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.fetch_timeout_s, connect=settings.connect_timeout_s),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            follow_redirects=True,
            max_redirects=3,
            event_hooks={"request": [self._guard_request]},
        )
        # Created lazily: only s3:// inputs need AWS, and client creation
        # itself can fail on machines without AWS configuration.
        self._s3: Any = None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def load(self, source: ImageSource) -> bytes:
        """Return decoded image bytes for a base64 or URL source."""
        if isinstance(source, Base64Image):
            return self._load_base64(source)
        if isinstance(source, UrlImage):
            return await self._load_url(source.url)
        assert_never(source)

    def _load_base64(self, source: Base64Image) -> bytes:
        data = base64.b64decode("".join(source.data.split()), validate=True)
        if len(data) > self._settings.max_image_bytes:
            raise ImageTooLargeError(
                f"Image exceeds the {self._settings.max_image_bytes} byte limit."
            )
        if not data:
            raise BadImageError("Base64 image data is empty.")
        return data

    async def _load_url(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme == "s3":
            url = await asyncio.to_thread(self._presign_s3, parsed.netloc, parsed.path.lstrip("/"))
            parsed = urlparse(url)
        await self._assert_fetchable_host(parsed.hostname or "")
        return await self._fetch_bounded(url)

    async def _guard_request(self, request: httpx.Request) -> None:
        """Host-check every hop before it is sent, redirect targets included."""
        await self._assert_fetchable_host(request.url.host)

    def _presign_s3(self, bucket: str, key: str) -> str:
        if not bucket or not key:
            raise BadImageError("S3 URL must include a bucket and key.")
        try:
            if self._s3 is None:
                self._s3 = boto3.client("s3")
            return str(
                self._s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=self._settings.s3_presign_ttl_s,
                )
            )
        except Exception as exc:
            logger.warning("s3 presign failed: bucket=%s key=%s err=%r", bucket, key, exc)
            raise ImageFetchError("Could not access the S3 object.") from exc

    async def _assert_fetchable_host(self, host: str) -> None:
        if not host:
            raise BadImageError("Image URL must include a host.")
        if self._settings.allow_private_hosts:
            return
        if _looks_private(host):
            raise BadImageError("Image URL host is not allowed.")
        if not await _resolve_is_global(host):
            raise BadImageError("Image URL host is not allowed.")

    async def _fetch_bounded(self, url: str) -> bytes:
        limit = self._settings.max_image_bytes
        try:
            # Total deadline: per-operation timeouts alone never fire against a
            # server that drips one chunk just inside every read timeout.
            async with asyncio.timeout(self._settings.fetch_timeout_s):
                async with self._client.stream("GET", url) as response:
                    response.raise_for_status()
                    declared = response.headers.get("content-length")
                    if declared is not None and declared.isdigit() and int(declared) > limit:
                        raise ImageTooLargeError(f"Image exceeds the {limit} byte limit.")
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        received += len(chunk)
                        if received > limit:
                            raise ImageTooLargeError(f"Image exceeds the {limit} byte limit.")
                        chunks.append(chunk)
        except TimeoutError:
            logger.warning("image fetch timed out: host=%s", urlparse(url).hostname)
            raise ImageFetchError("Could not download the image.") from None
        except httpx.HTTPStatusError as exc:
            logger.warning("image fetch status error: status=%s", exc.response.status_code)
            raise ImageFetchError(f"Image URL returned status {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            logger.warning("image fetch failed: err=%r", exc)
            raise ImageFetchError("Could not download the image.") from exc
        data = b"".join(chunks)
        if not data:
            raise BadImageError("Downloaded image is empty.")
        return data
