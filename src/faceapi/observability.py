"""Request IDs, structured logging, and tracing hooks.

Every response carries ``X-Request-ID``; log records from the ingress tier
include the same ID. Inference-tier batches serve multiple requests per
execution, so their logs are not request-scoped.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

_request_id_ctx: ContextVar[str] = ContextVar("faceapi_request_id", default="-")


def get_request_id() -> str:
    """Return the request ID for the current async context."""
    return _request_id_ctx.get()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Accept or mint a request ID and expose it to handlers and logs."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = _request_id_ctx.set(request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx.reset(token)
        # Error handlers already set the header; don't append a duplicate.
        if REQUEST_ID_HEADER not in response.headers:
            response.headers[REQUEST_ID_HEADER] = request_id
        return response


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": _request_id_ctx.get(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Install the JSON log handler on the ``faceapi`` logger exactly once.

    Only this package's logger is configured; the root logger and its
    handlers (Ray's, uvicorn's) are left untouched.
    """
    faceapi = logging.getLogger("faceapi")
    if any(
        isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter)
        for h in faceapi.handlers
    ):
        faceapi.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    faceapi.handlers.clear()
    faceapi.addHandler(handler)
    faceapi.setLevel(level)
    faceapi.propagate = False
