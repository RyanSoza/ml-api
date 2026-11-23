.PHONY: install lint format test train run docker-build docker-run

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

train:
	uv run python -m ml_api.train

run:
	uv run uvicorn ml_api.app:app --reload

docker-build:
	docker build -t ml-api:dev .

docker-run:
	docker run --rm -p 8000:8000 -e VERSION=local-docker ml-api:dev