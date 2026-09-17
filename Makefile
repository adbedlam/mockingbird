install:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run black --check --diff .
	uv run isort --check-only --diff .
	uv run mypy .

format:
	uv run black .
	uv run isort .

hooks:
	uv run pre-commit install

dev:
	install hooks