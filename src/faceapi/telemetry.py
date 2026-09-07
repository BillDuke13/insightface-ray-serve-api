"""Tracing setup and per-stage latency helpers.

Tracing uses OpenTelemetry and stays a no-op until ``FACEAPI_OTEL_ENABLED``
is true, in which case spans export via OTLP (endpoint and credentials come
from the standard ``OTEL_*`` environment variables). Deployment-level
request metrics (RPS, latency, errors) are provided by Ray Serve itself.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

from faceapi.config import Settings

_TRACER_NAME = "faceapi"
_initialized = False


def init_telemetry(settings: Settings) -> None:
    """Install the OTLP tracer provider exactly once when enabled."""
    global _initialized
    if not settings.otel_enabled or _initialized:
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


@contextmanager
def start_span(name: str) -> Iterator[None]:
    """Open a span when tracing is initialized, else do nothing."""
    if not _initialized:
        with nullcontext():
            yield
        return
    from opentelemetry import trace

    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name):
        yield


class StageTimer:
    """Accumulate per-stage latencies for one request."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._stages: dict[str, float] = {}

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self._stages[stage] = (now - self._started) * 1000
        self._started = now

    def summary(self) -> str:
        return " ".join(f"{name}={ms:.1f}ms" for name, ms in self._stages.items())
