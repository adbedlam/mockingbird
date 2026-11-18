.PHONY: install test lint format hooks dev

install:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run black --check --diff src scripts tests
	uv run isort --check-only --diff src scripts tests
	uv run mypy

format:
	uv run isort src scripts tests
	uv run black src scripts tests

hooks:
	uv run pre-commit install

dev: install hooks
