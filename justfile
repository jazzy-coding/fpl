set shell := ["powershell", "-Command"]

install:
	uv sync --frozen

lint:
	uv run ruff check .

test:
	uv run pytest -vv --cov=tests tests

ci: install lint test
