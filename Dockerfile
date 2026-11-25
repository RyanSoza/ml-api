# ---------- Stage 1: build the virtual environment ----------
FROM python:3.12-slim AS builder

# Borrow the uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

# 1. Dependencies
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. The code
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---------- Stage 2: Image shipped ----------
FROM python:3.12-slim

RUN useradd --create-home app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH"
ARG GIT_SHA=dev
ENV VERSION=$GIT_SHA
USER app
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn ml_api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]