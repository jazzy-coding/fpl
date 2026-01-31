set shell := ["powershell", "-Command"]

install:
	uv sync --frozen

lint:
	uv run ruff check .

format:
	uv run black .

test:
	uv run pytest -vv --cov=tests tests

ci: install lint format test
