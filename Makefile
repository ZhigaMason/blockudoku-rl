# Common tasks. Everything goes through uv (Python) and npm (dev-only JS test deps).
RUN ?= runs/small
CONFIG ?= small
PORT ?= 8000

.PHONY: install test lint sync check train eval baselines export serve

install:            ## Python + JS dev dependencies
	uv sync
	npm ci

test:               ## full test suite (Python, JS parity, ONNX/onnxruntime-web parity)
	uv run pytest -q

lint:
	uv run ruff check src tests

sync:               ## regenerate web/pieces.json + docs/RULES.md appendix from pieces.py
	uv run python -m blockudoku.sync_assets

check: lint test    ## what CI runs
	uv run python -m blockudoku.sync_assets --check

train:              ## make train CONFIG=small RUN=runs/small
	uv run python -m blockudoku.train --config $(CONFIG) --out $(RUN)

eval:               ## make eval RUN=runs/small
	uv run python -m blockudoku.evaluate $(RUN)

baselines:
	uv run python -m blockudoku.evaluate --baseline random
	uv run python -m blockudoku.evaluate --baseline greedy

export:             ## make export RUN=runs/small  -> web/model/
	uv run python -m blockudoku.export_onnx $(RUN)

serve:              ## play at http://127.0.0.1:$(PORT)/
	cd web && python3 -m http.server $(PORT) --bind 127.0.0.1
