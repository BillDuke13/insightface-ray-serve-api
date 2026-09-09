"""Domain errors and their mapping to HTTP responses.

Every error returned to callers carries ``code``/``message``/``request_id``.
Internal exception details never leave the process: unexpected failures are
logged server-side and surface as a generic 500.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from faceapi.observability import REQUEST_ID_HEADER, get_request_id

logger = logging.getLogger(__name__)


def _request_id_from(request: Request) -> str:
    """Prefer the ID stored on the request scope; it survives re-raises."""
    stored = getattr(request.state, "request_id", None)
    if isinstance(stored, str) and stored:
        return stored
    return get_request_id()


class FaceAPIError(Exception):
    """Base class for expected, caller-facing failures."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadImageError(FaceAPIError):
    """The image bytes cannot be decoded or are not an image."""

    status_code = 400
    code = "BAD_IMAGE"


class ImageTooLargeError(FaceAPIError):
    """The image exceeds the configured size limit."""

    status_code = 413
    code = "IMAGE_TOO_LARGE"


class ImageFetchError(FaceAPIError):
    """Downloading the image from its URL failed."""

    status_code = 502
    code = "IMAGE_FETCH_FAILED"


class NoFaceError(FaceAPIError):
    """A face was required but none was detected."""

    status_code = 422
    code = "NO_FACE_FOUND"


class LowQualityError(FaceAPIError):
    """The detected face is below the quality gate."""

    status_code = 422
    code = "LOW_QUALITY_FACE"


class OverloadedError(FaceAPIError):
    """The ingress concurrency guard rejected the request."""

    status_code = 429
    code = "OVERLOADED"


class UnavailableError(FaceAPIError):
    """A dependency (e.g. inference replicas) is not ready yet."""

    status_code = 503
    code = "NOT_READY"


def build_error_body(code: str, message: str, request_id: str) -> dict[str, str]:
    return {"code": code, "message": message, "request_id": request_id}


def _truncate(value: str, limit: int = 300) -> str:
    return value if len(value) <= limit else value[:limit] + "...(truncated)"


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain errors, validation errors, and bugs to JSON responses."""

    @app.exception_handler(FaceAPIError)
    async def handle_faceapi_error(request: Request, exc: FaceAPIError) -> JSONResponse:
        request_id = _request_id_from(request)
        logger.warning(
            "request failed: code=%s status=%s msg=%s path=%s",
            exc.code,
            exc.status_code,
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_body(exc.code, exc.message, request_id),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = _request_id_from(request)
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        logger.warning("validation failed: path=%s details=%s", request.url.path, details)
        return JSONResponse(
            status_code=400,
            content=build_error_body("VALIDATION_ERROR", _truncate(details), request_id),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id_from(request)
        logger.exception("unexpected error: path=%s exc=%r", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content=build_error_body("INTERNAL_ERROR", "Internal server error.", request_id),
            headers={REQUEST_ID_HEADER: request_id},
        )
