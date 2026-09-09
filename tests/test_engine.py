"""Engine tests with a fake InspireFace session (no native models)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from faceapi.config import Settings
from faceapi.engine import FaceEngine
from faceapi.errors import LowQualityError, NoFaceError


def _face(
    box: tuple[int, int, int, int] = (10, 10, 50, 50),
    confidence: float = 0.9,
) -> SimpleNamespace:
    return SimpleNamespace(
        location=list(box),
        detection_confidence=confidence,
        roll=1.0,
        yaw=2.0,
        pitch=3.0,
    )


def _ext(
    quality: float = 0.9,
    gender: Any = 1,
    age_bracket: Any = 3,
    race: Any = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        quality_confidence=quality,
        mask_confidence=0.01,
        rgb_liveness_confidence=0.95,
        gender=gender,
        age_bracket=age_bracket,
        race=race,
    )


class FakeSession:
    """Configurable stand-in for the native session."""

    def __init__(
        self,
        faces: list[SimpleNamespace] | None = None,
        exts: list[SimpleNamespace] | None = None,
        feature: npt.NDArray[Any] | None = None,
        landmarks: list[tuple[int, int]] | None = None,
    ) -> None:
        self._faces = faces if faces is not None else [_face()]
        self._exts = exts if exts is not None else [_ext()]
        self._feature: npt.NDArray[Any] | None = (
            np.zeros(4, dtype=np.float32) if feature is None else feature
        )
        self._landmarks = landmarks if landmarks is not None else [(1, 2), (3, 4)]
        self.pipeline_flags = -1

    def face_detection(self, image: npt.NDArray[Any]) -> list[Any]:
        assert image.ndim == 3
        return list(self._faces)

    def face_pipeline(self, image: npt.NDArray[Any], faces: list[Any], flags: int) -> list[Any]:
        self.pipeline_flags = flags
        return list(self._exts)

    def face_feature_extract(self, image: npt.NDArray[Any], face: Any) -> npt.NDArray[Any] | None:
        return self._feature

    def get_face_dense_landmark(self, face: Any) -> Any:
        return list(self._landmarks)


def _image() -> npt.NDArray[Any]:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _engine(**overrides: Any) -> FaceEngine:
    return FaceEngine(Settings(**overrides), FakeSession(), FakeSession())


def test_detect_empty() -> None:
    engine = FaceEngine(Settings(), FakeSession(faces=[]), FakeSession(faces=[]))
    assert engine.detect(_image()) == []


def test_detect_sorts_and_caps() -> None:
    faces = [_face(confidence=0.6), _face(confidence=0.95), _face(confidence=0.7)]
    session = FakeSession(faces=faces, exts=[_ext(), _ext(), _ext()])
    engine = FaceEngine(Settings(), session, FakeSession())
    results = engine.detect(_image(), max_faces=2)
    assert [r.confidence for r in results] == [0.95, 0.7]


def test_detect_maps_attributes() -> None:
    results = _engine().detect(_image(), include_embedding=True)
    assert len(results) == 1
    face = results[0]
    assert face.bounding_box == (10, 10, 50, 50)
    assert face.gender == "Male"
    assert face.age_bracket == "20-29 years old"
    assert face.race == "Asian"
    assert face.embedding == [0.0, 0.0, 0.0, 0.0]
    assert face.landmarks == [(1, 2), (3, 4)]
    assert face.landmarks_truncated is False


def test_detect_truncates_landmarks() -> None:
    session = FakeSession(landmarks=[(i, i) for i in range(200)])
    engine = FaceEngine(Settings(max_landmarks=10), session, FakeSession())
    (face,) = engine.detect(_image())
    assert len(face.landmarks or []) == 10
    assert face.landmarks_truncated is True


def test_detect_handles_numpy_landmarks() -> None:
    session = FakeSession(landmarks=np.array([[1, 2], [3, 4]]))  # type: ignore[arg-type]
    engine = FaceEngine(Settings(), session, FakeSession())
    (face,) = engine.detect(_image())
    assert face.landmarks == [(1, 2), (3, 4)]


def test_detect_skips_landmarks_and_embedding_by_default() -> None:
    (face,) = _engine().detect(_image(), include_landmarks=False)
    assert face.landmarks is None
    assert face.embedding is None


def test_detect_failed_embedding_is_none() -> None:
    session = FakeSession(feature=None)
    session._feature = None  # Force extraction failure after init default.
    engine = FaceEngine(Settings(), session, FakeSession())
    (face,) = engine.detect(_image(), include_embedding=True)
    assert face.embedding is None


def test_detect_rejects_short_bbox() -> None:
    session = FakeSession(faces=[_face(box=(1, 2))])  # type: ignore[arg-type]
    engine = FaceEngine(Settings(), session, FakeSession())
    with pytest.raises(ValueError):
        engine.detect(_image())


@pytest.mark.parametrize(
    ("gender", "age", "race", "expected"),
    [(9, 3, 1, (None, "20-29 years old", "Asian")), (1, -1, 99, ("Male", None, None))],
)
def test_detect_out_of_range_tags_are_none(
    gender: Any, age: Any, race: Any, expected: tuple[Any, Any, Any]
) -> None:
    session = FakeSession(exts=[_ext(gender=gender, age_bracket=age, race=race)])
    engine = FaceEngine(Settings(), session, FakeSession())
    (face,) = engine.detect(_image())
    assert (face.gender, face.age_bracket, face.race) == expected


def test_compare_picks_largest_face(monkeypatch: pytest.MonkeyPatch) -> None:
    small, big = _face(box=(0, 0, 10, 10)), _face(box=(0, 0, 100, 100))
    seen: list[Any] = []

    class SpySession(FakeSession):
        def face_feature_extract(
            self, image: npt.NDArray[Any], face: Any
        ) -> npt.NDArray[Any] | None:
            seen.append(face)
            return np.ones(4, dtype=np.float32)

    engine = FaceEngine(Settings(), FakeSession(), SpySession(faces=[small, big]))
    monkeypatch.setattr("faceapi.engine.isf.feature_comparison", lambda a, b: 0.77)
    assert engine.compare(_image(), _image()) == pytest.approx(0.77)
    assert seen and all(face is big for face in seen)


def test_detect_caps_at_settings_even_when_request_is_higher() -> None:
    faces = [_face(confidence=0.9), _face(confidence=0.8)]
    session = FakeSession(faces=faces, exts=[_ext(), _ext()])
    engine = FaceEngine(Settings(max_faces=1), session, FakeSession())
    assert len(engine.detect(_image(), max_faces=200)) == 1


def test_compare_rejects_tiny_primary_face() -> None:
    session = FakeSession(faces=[_face(box=(0, 0, 10, 10))])
    engine = FaceEngine(Settings(), FakeSession(), session)
    with pytest.raises(LowQualityError):
        engine.compare(_image(), _image())


def test_compare_no_face() -> None:
    engine = FaceEngine(Settings(), FakeSession(), FakeSession(faces=[]))
    with pytest.raises(NoFaceError):
        engine.compare(_image(), _image())


def test_compare_failed_feature() -> None:
    session = FakeSession()
    session._feature = None
    engine = FaceEngine(Settings(), FakeSession(), session)
    with pytest.raises(NoFaceError):
        engine.compare(_image(), _image())


def test_detect_maps_sdk_sentinel_scores_to_none() -> None:
    """Uncomputed SDK scores (-1) surface as null, never as -1.0."""
    ext = SimpleNamespace(
        quality_confidence=-1.0,
        mask_confidence=-1.0,
        rgb_liveness_confidence=-1,
        gender=1,
        age_bracket=3,
        race=1,
    )
    session = FakeSession(exts=[ext])
    engine = FaceEngine(Settings(), session, FakeSession())
    (face,) = engine.detect(_image())
    assert face.quality is None
    assert face.mask_confidence is None
    assert face.liveness_confidence is None
