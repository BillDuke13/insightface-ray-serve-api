"""Telemetry helper tests (tracing disabled by default)."""

from __future__ import annotations

from faceapi.config import Settings
from faceapi.telemetry import StageTimer, init_telemetry, start_span


def test_span_without_init_is_noop() -> None:
    init_telemetry(Settings(otel_enabled=False))
    with start_span("noop"):
        pass


def test_stage_timer_summary() -> None:
    timer = StageTimer()
    timer.mark("fetch_ms")
    timer.mark("decode_ms")
    summary = timer.summary()
    assert "fetch_ms=" in summary and "decode_ms=" in summary
