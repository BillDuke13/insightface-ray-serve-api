# AGENTS.md — insightface-ray-serve-api

Face detection + 1:1 comparison REST API on Ray Serve
(two tiers: `FaceIngress` on CPU, `FaceInference` on CPU/GPU).

## Environment

uv-managed: `uv venv --python 3.11 && uv pip install -e ".[dev]"`,
then `serve run serve-cpu.yaml`. Conda was removed; do not reintroduce it.

## Gates (must stay green)

`ruff check`, `ruff format --check`, and `mypy` over
`src/faceapi tests scripts`, then `pytest`. CI runs the same gates.

## Constraints

- FastAPI must stay at 0.139.1 (see the `pyproject.toml` comment);
  `serve_app.py` must not use `from __future__ import annotations`
  (see its module docstring).
- Docs, docstrings, and code ship together: a behavior change updates
  the README / `docs/api.md` / affected docstrings in the same change.
- Keep modules small (guide: ≤300 lines, functions ≤60); no commented-out code.
