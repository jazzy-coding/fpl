.PHONY: install 

$ make install
uv sync

$ make lint
ruff check .

$ make test
python3 -m pytest -vv --cov=tests/ tests/
