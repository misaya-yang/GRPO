.PHONY: setup test lint format exact smoke paper

PYTHON ?= python
RUFF ?= ruff
RUN = PYTHONPATH=src $(PYTHON)

setup:
	uv sync --locked

test:
	$(RUN) -m pytest -q

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .

exact:
	$(RUN) scripts/validate_exact.py --output runs/exact/$$(date -u +%Y%m%dT%H%M%SZ).json

smoke:
	$(RUN) -m pytest tests/test_llm.py tests/test_feedback_llm.py -q

paper:
	cd paper && latexmk -pdf -outdir=build main.tex
