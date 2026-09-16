install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

dev:
	uv sync
	uv run pre-commit install

hooks:
	uv run pre-commit install
