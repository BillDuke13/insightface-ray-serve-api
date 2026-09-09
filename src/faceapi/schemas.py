"""Pydantic v2 request/response schemas for the face API.

Breaking v2 contract: the image source is a discriminated union, so exactly
one of base64/URL is enforced by the type system instead of a root validator.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

ALLOWED_URL_SCHEMES = ("http", "https", "s3")


class Base64Image(BaseModel):
    """Image bytes carried inline as a base64 string."""

    type: Literal["base64"] = "base64"
    data: str = Field(..., description="Base64-encoded image bytes.")

    @field_validator("data")
    @classmethod
    def _check_base64(cls, value: str) -> str:
        cleaned = "".join(value.split())
        if not cleaned:
            raise ValueError("Base64 image data must not be empty.")
        try:
            base64.b64decode(cleaned, validate=True)
        except binascii.Error as exc:
            raise ValueError("Invalid base64 image data.") from exc
        return value


class UrlImage(BaseModel):
    """Image fetched by the server from an http/https/s3 URL."""

    type: Literal["url"] = "url"
    url: str = Field(..., description="http, https, or s3 URL of the image.")

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            raise ValueError(f"URL scheme must be one of {ALLOWED_URL_SCHEMES}.")
        if not parsed.netloc:
            raise ValueError("URL must include a host.")
        return value


ImageSource = Annotated[Base64Image | UrlImage, Field(discriminator="type")]


class DetectRequest(BaseModel):
    """Face detection request with response-shape switches."""

    image: ImageSource
    include_embedding: bool = Field(
        default=False, description="Include the face embedding vector when true."
    )
    include_landmarks: bool = Field(
        default=True, description="Include facial landmark coordinates when true."
    )
    max_faces: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Cap on faces returned, newest API default otherwise.",
    )


class FaceResult(BaseModel):
    """One detected face with geometry, pose, quality, and attributes."""

    bounding_box: tuple[int, int, int, int] = Field(..., description="(x1, y1, x2, y2) in pixels.")
    confidence: float = Field(..., description="Detection confidence score.")
    landmarks: list[tuple[int, int]] | None = Field(
        default=None, description="Facial landmark coordinates, truncated to the cap."
    )
    landmarks_truncated: bool = Field(default=False, description="True when landmarks were capped.")
    embedding: list[float] | None = Field(default=None, description="Face embedding vector.")
    roll: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    quality: float | None = Field(default=None, description="Face quality score.")
    mask_confidence: float | None = None
    liveness_confidence: float | None = None
    gender: str | None = None
    age_bracket: str | None = None
    race: str | None = None


class DetectionResponse(BaseModel):
    """Detection response: every face found, highest confidence first."""

    faces: list[FaceResult]


class CompareRequest(BaseModel):
    """1:1 face comparison between the primary face of two images."""

    image1: ImageSource
    image2: ImageSource


class ComparisonResponse(BaseModel):
    """Cosine similarity of the two primary faces, higher means more similar."""

    similarity: float


class ErrorBody(BaseModel):
    """Uniform error envelope for every 4xx/5xx response."""

    code: str
    message: str
    request_id: str
