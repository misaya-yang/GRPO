.PHONY: setup test lint format exact smoke paper

setup:
	uv sync --locked

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

exact:
	uv run dependent-rollouts exact --output runs/exact/$$(date -u +%Y%m%dT%H%M%SZ).json

smoke:
	uv run --extra llm pytest tests/test_llm.py -q

paper:
	cd paper && latexmk -pdf -outdir=build main.tex
