"""Synchronous face engine wrapping InspireFace sessions.

Two sessions are kept: a full-capability session for detection and a
recognition-only session for comparison. Everything here is blocking;
callers run it in worker threads, never on the event loop. The
:class:`FaceSession` protocol keeps the engine unit-testable with a fake
session instead of native models.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

import inspireface as isf
import numpy.typing as npt

from faceapi.config import Settings
from faceapi.errors import LowQualityError, NoFaceError
from faceapi.schemas import FaceResult

logger = logging.getLogger(__name__)

GENDER_TAGS = ("Female", "Male")
AGE_BRACKET_TAGS = (
    "0-2 years old",
    "3-9 years old",
    "10-19 years old",
    "20-29 years old",
    "30-39 years old",
    "40-49 years old",
    "50-59 years old",
    "60-69 years old",
    "more than 70 years old",
)
RACE_TAGS = ("Black", "Asian", "Latino/Hispanic", "Middle Eastern", "White")

_FULL_CAPABILITIES = (
    isf.HF_ENABLE_FACE_RECOGNITION
    | isf.HF_ENABLE_QUALITY
    | isf.HF_ENABLE_MASK_DETECT
    | isf.HF_ENABLE_LIVENESS
    | isf.HF_ENABLE_INTERACTION
    | isf.HF_ENABLE_FACE_ATTRIBUTE
)
_RECOGNITION_ONLY = isf.HF_ENABLE_FACE_RECOGNITION


class FaceSession(Protocol):
    """Minimal InspireFace session surface used by the engine."""

    def face_detection(self, image: npt.NDArray[Any]) -> list[Any]: ...
    def face_pipeline(self, image: npt.NDArray[Any], faces: list[Any], flags: int) -> list[Any]: ...
    def face_feature_extract(
        self, image: npt.NDArray[Any], face: Any
    ) -> npt.NDArray[Any] | None: ...
    def get_face_dense_landmark(self, face: Any) -> Any: ...


def _open_session(capabilities: int, detection_threshold: float) -> Any:
    session = isf.InspireFaceSession(capabilities, isf.HF_DETECT_MODE_ALWAYS_DETECT)
    session.set_detection_confidence_threshold(detection_threshold)
    return session


def _tag(tags: tuple[str, ...], index: Any) -> str | None:
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return tags[index] if 0 <= index < len(tags) else None


def _bbox(face: Any) -> tuple[int, int, int, int]:
    loc = list(face.location)
    if len(loc) < 4:
        raise ValueError(f"Face location has {len(loc)} coordinates, expected 4.")
    return (int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3]))


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


class FaceEngine:
    """Detect faces and compare primary faces across images."""

    def __init__(
        self,
        settings: Settings,
        detect_session: FaceSession | None = None,
        compare_session: FaceSession | None = None,
    ) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._detect_session: Any = detect_session or _open_session(
            _FULL_CAPABILITIES, settings.detection_threshold
        )
        self._compare_session: Any = compare_session or _open_session(
            _RECOGNITION_ONLY, settings.detection_threshold
        )

    def detect(
        self,
        image: npt.NDArray[Any],
        *,
        include_embedding: bool = False,
        include_landmarks: bool = True,
        max_faces: int | None = None,
    ) -> list[FaceResult]:
        """Detect faces and extract attributes, highest confidence first."""
        with self._lock:
            faces = self._detect_session.face_detection(image)
            if not faces:
                return []
            extends = self._detect_session.face_pipeline(image, faces, _FULL_CAPABILITIES)
            scored = sorted(
                zip(faces, extends, strict=True),
                key=lambda pair: float(pair[0].detection_confidence),
                reverse=True,
            )
            if max_faces is None:
                cap = self._settings.max_faces
            else:
                cap = min(max_faces, self._settings.max_faces)
            results = [
                self._format(image, face, ext, include_embedding, include_landmarks)
                for face, ext in scored[:cap]
            ]
        logger.debug("detected faces: count=%d", len(results))
        return results

    def compare(self, image1: npt.NDArray[Any], image2: npt.NDArray[Any]) -> float:
        """Return cosine similarity of the primary face in each image."""
        feature1 = self._primary_feature(image1, "image1")
        feature2 = self._primary_feature(image2, "image2")
        similarity = float(isf.feature_comparison(feature1, feature2))
        logger.debug("compared faces: similarity=%.4f", similarity)
        return similarity

    def _primary_feature(self, image: npt.NDArray[Any], label: str) -> npt.NDArray[Any]:
        with self._lock:
            faces = self._compare_session.face_detection(image)
            if not faces:
                raise NoFaceError(f"No face detected in {label}.")
            primary = max(faces, key=lambda face: _area(_bbox(face)))
            if _area(_bbox(primary)) < self._settings.min_face_pixels:
                raise LowQualityError(f"Primary face in {label} is too small to compare reliably.")
            feature = self._compare_session.face_feature_extract(image, primary)
        if feature is None:
            raise NoFaceError(f"Could not extract a face feature from {label}.")
        result: npt.NDArray[Any] = feature
        return result

    def _format(
        self,
        image: npt.NDArray[Any],
        face: Any,
        ext: Any,
        include_embedding: bool,
        include_landmarks: bool,
    ) -> FaceResult:
        box = _bbox(face)
        landmarks: list[tuple[int, int]] | None = None
        truncated = False
        if include_landmarks:
            raw = self._detect_session.get_face_dense_landmark(face)
            points = (
                [(int(lm[0]), int(lm[1])) for lm in raw if len(lm) >= 2] if raw is not None else []
            )
            cap = self._settings.max_landmarks
            landmarks = points[:cap]
            truncated = len(points) > cap
        embedding: list[float] | None = None
        if include_embedding:
            vector = self._detect_session.face_feature_extract(image, face)
            if vector is not None:
                embedding = [float(v) for v in vector.tolist()]
            else:
                logger.warning("embedding extraction failed: box=%s", box)
        return FaceResult(
            bounding_box=box,
            confidence=float(face.detection_confidence),
            landmarks=landmarks,
            landmarks_truncated=truncated,
            embedding=embedding,
            roll=self._optional_float(face, "roll"),
            yaw=self._optional_float(face, "yaw"),
            pitch=self._optional_float(face, "pitch"),
            quality=self._score(ext, "quality_confidence"),
            mask_confidence=self._score(ext, "mask_confidence"),
            liveness_confidence=self._score(ext, "rgb_liveness_confidence"),
            gender=_tag(GENDER_TAGS, getattr(ext, "gender", None)),
            age_bracket=_tag(AGE_BRACKET_TAGS, getattr(ext, "age_bracket", None)),
            race=_tag(RACE_TAGS, getattr(ext, "race", None)),
        )

    @staticmethod
    def _optional_float(obj: Any, name: str) -> float | None:
        value = getattr(obj, name, None)
        return None if value is None else float(value)

    @staticmethod
    def _score(obj: Any, name: str) -> float | None:
        """Read a confidence score; the SDK's -1 'uncomputed' sentinel maps to None."""
        value = getattr(obj, name, None)
        if value is None:
            return None
        score = float(value)
        return None if score < 0 else score

    def assert_usable(self) -> None:
        """Raise when either native session failed to initialize."""
        if self._detect_session is None or self._compare_session is None:
            raise RuntimeError("Face engine sessions are not initialized.")
